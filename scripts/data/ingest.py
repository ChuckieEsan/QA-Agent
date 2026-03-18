import os
import sys
import sqlite3
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

sys.path.append(os.getcwd())

from src.config.setting import settings
from src.app.infra.utils import generate_doc_id, get_device, clean_text
from src.app.infra.utils.logger import get_logger
from src.app.infra.db.milvus_db import MilvusDBClient
from src.app.infra.db.postgres_db import PostgresDBClient

logger = get_logger(__name__)

# ================= 配置区域 =================
BATCH_SIZE = 16
MODEL_PATH = str(settings.models.embedding_model_path)
MILVUS_DB_PATH = str(settings.milvus_db.db_path)
GOV_CASES_COLLECTION_NAME = (
    settings.vectordb.gov_cases_collection_name
)  # 问政案例 collection/table
GOV_POWERS_COLLECTION_NAME = (
    settings.vectordb.gov_powers_collection_name
)  # 行政权力清单 collection/table
SQLITE_DB_PATH = str(settings.paths.raw_data_db_path)
DATA_DIR = settings.paths.data_dir


def get_db_client(db_type: str):
    """
    根据 db_type 获取数据库客户端

    Args:
        db_type: 数据库类型 (milvus/postgres)

    Returns:
        数据库客户端实例
    """
    if db_type == "postgres":
        return PostgresDBClient()
    else:
        return MilvusDBClient()


def init_milvus(
    client: MilvusDBClient, collection_name: str, drop_existing: bool = False
):
    """初始化 Milvus 集合 Schema"""
    if client.has_collection(collection_name):
        if drop_existing:
            logger.info(f"检测到集合 {collection_name} 已存在，正在删除重建...")
            client.drop_collection(collection_name)
        else:
            logger.info(f"集合 {collection_name} 已存在，将使用增量模式...")
            return False  # 集合已存在，返回 False 表示不需要重建

    logger.info(f"创建新集合 {collection_name} Schema...")
    client._client.create_collection(
        collection_name=collection_name,
        dimension=settings.models.embedding_size,  # BGE-M3 维度
        metric_type="COSINE",
        auto_id=True,
        enable_dynamic_field=True,
    )
    return True  # 集合新建成功


def init_postgres(
    client: PostgresDBClient, table_name: str, drop_existing: bool = False
):
    """初始化 PostgreSQL 表"""
    if client.has_collection(table_name):
        if drop_existing:
            logger.info(f"检测到表 {table_name} 已存在，正在删除重建...")
            client.drop_collection(table_name)
        else:
            logger.info(f"表 {table_name} 已存在，将使用增量模式...")
            return False
    return True


def fetch_data_from_sqlite(db_path: str):
    """从 SQLite 读取所有已爬取的数据"""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"找不到数据库文件: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # 允许通过列名访问
    cursor = conn.cursor()

    # 仅读取有标题、问题和回复的数据
    cursor.execute(
        """
        SELECT id, title, dept, question, answer, question_time, url
        FROM wenzheng
        WHERE question IS NOT NULL AND answer IS NOT NULL
    """
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def fetch_existing_doc_ids_milvus(client: MilvusDBClient, collection_name: str) -> set:
    """从 Milvus 获取已存在的 doc_id 集合"""
    try:
        # 查询所有数据的 metadata.doc_id 字段
        results = client.query(
            collection_name=collection_name,
            output_fields=["metadata.doc_id"],
        )
        doc_ids = set()
        for item in results:
            if "metadata" in item and "doc_id" in item.get("metadata", {}):
                doc_ids.add(item["metadata"]["doc_id"])

        logger.info(f"已获取 {len(doc_ids)} 条已存在的 doc_id")
        return doc_ids
    except Exception as e:
        logger.warning(f"查询已存在数据失败: {e}，将执行全量导入")
        return set()


def fetch_existing_doc_ids_postgres(client: PostgresDBClient, table_name: str) -> set:
    """从 PostgreSQL 获取已存在的 doc_id 集合"""
    try:
        # 通过 JSONB 查询 metadata->>'doc_id'
        results = client.query(
            collection_name=table_name,
            filter_expr="metadata IS NOT NULL",
            output_fields=["metadata"],
        )
        doc_ids = set()
        for item in results:
            metadata = item.get("metadata", {})
            if isinstance(metadata, dict) and "doc_id" in metadata:
                doc_ids.add(metadata["doc_id"])

        logger.info(f"已获取 {len(doc_ids)} 条已存在的 doc_id")
        return doc_ids
    except Exception as e:
        logger.warning(f"查询已存在数据失败: {e}，将执行全量导入")
        return set()


def fetch_existing_doc_ids(client, collection_name: str, db_type: str) -> set:
    """获取已存在的 doc_id 集合（根据 db_type 选择实现）"""
    if db_type == "postgres":
        return fetch_existing_doc_ids_postgres(client, collection_name)
    else:
        return fetch_existing_doc_ids_milvus(client, collection_name)


def ingest_cases(client, embed_model: SentenceTransformer, db_type: str = "milvus"):
    """导入问政案例数据（全量导入）"""
    # 1. 读取数据
    try:
        rows = fetch_data_from_sqlite(SQLITE_DB_PATH)
    except Exception as e:
        logger.error(f"读取 SQLite 失败: {e}")
        return

    logger.info(f"有效数据量: {len(rows)} 条")
    if len(rows) == 0:
        logger.warning("数据库为空，请先运行 crawl.py")
        return

    # 2. 删除已有数据并重建表
    logger.info(f"删除已有表 {GOV_CASES_COLLECTION_NAME} 并重建...")
    if db_type == "postgres":
        client.drop_collection(GOV_CASES_COLLECTION_NAME)
        init_postgres(client, GOV_CASES_COLLECTION_NAME)
    else:
        client.drop_collection(GOV_CASES_COLLECTION_NAME)
        init_milvus(client, GOV_CASES_COLLECTION_NAME)

    # 3. 批量处理
    data_list = [dict(row) for row in rows]
    total_rows = len(data_list)
    logger.info(f"开始向量化并入库...（共 {total_rows} 条）")

    for i in tqdm(range(0, total_rows, BATCH_SIZE), desc="Processing cases"):
        batch = data_list[i : i + BATCH_SIZE]

        # 准备 Embedding 的文本：主要使用"问题"
        texts_to_embed = [clean_text(item["question"]) for item in batch]

        # 生成向量
        vectors = embed_model.encode(texts_to_embed, normalize_embeddings=True)

        data_to_insert = []
        for j, item in enumerate(batch):
            # 清洗文本
            question_text = clean_text(item["question"])
            answer_text = clean_text(item["answer"])
            title_text = item["title"]
            dept_text = item["dept"]
            time_text = item["question_time"]
            url_text = item["url"]

            doc_id = generate_doc_id(question_text, answer_text)

            # === 构建 RAG 上下文 (Rich Context) ===
            rag_context = (
                f"标题：{title_text}\n"
                f"部门：{dept_text}\n"
                f"时间：{time_text}\n"
                f"市民诉求：{question_text}\n"
                f"官方回复：{answer_text}\n"
                f"来源链接：{url_text}"
            )

            data_to_insert.append(
                {
                    "vector": vectors[j],
                    "text": rag_context,
                    "department": dept_text,
                    "title": title_text,
                    "question": question_text,
                    "answer": answer_text,
                    "doc_type": "gov_case",  # 区分于行政权力清单
                    "metadata": {
                        "doc_id": doc_id,
                        "crawler_id": item["id"],  # 原始ID
                        "url": url_text,
                        "time": time_text,
                    },
                }
            )

        client.insert(GOV_CASES_COLLECTION_NAME, data_to_insert)

    logger.info(f"问政案例入库完成！共导入 {total_rows} 条数据")


def ingest_gov_powers(
    client, embed_model: SentenceTransformer, db_type: str = "milvus"
):
    """导入行政权力清单数据"""
    excel_file = DATA_DIR / "泸州市市本级行政权力清单目录（2021年本）.xlsx"

    if not excel_file.exists():
        logger.error(f"找不到行政权力清单文件: {excel_file}")
        return

    # 初始化数据库
    logger.info(f"删除已有表 {GOV_POWERS_COLLECTION_NAME} 并重建...")
    if db_type == "postgres":
        client.drop_collection(GOV_POWERS_COLLECTION_NAME)
        init_postgres(client, GOV_POWERS_COLLECTION_NAME)
    else:
        client.drop_collection(GOV_POWERS_COLLECTION_NAME)
        init_milvus(client, GOV_POWERS_COLLECTION_NAME)

    logger.info(f"读取行政权力清单: {excel_file}")
    guide_data_file = pd.ExcelFile(excel_file)
    sheet_names = guide_data_file.sheet_names

    all_documents = []

    for id, sheet_name in enumerate(sheet_names):
        logger.info(f"处理 Sheet: {sheet_name}")
        df = guide_data_file.parse(
            sheet_name=sheet_name, index_col=0, skiprows=2 if id == 0 else 1
        )

        # 构建 semantic_text
        df["semantic_text"] = df.apply(
            lambda row: f"【{row['市级部门']}】的法定职责包含{row['权力类型']}：{row['权力名称']}。{row['备注'] if pd.notna(row['备注']) else ''}",
            axis=1,
        )

        for _, row in df.iterrows():
            # 处理备注字段的 NaN 值
            note = row["备注"] if pd.notna(row["备注"]) else None
            metadata = {"note": note} if note is not None else {}

            all_documents.append(
                {
                    "text": row["semantic_text"],
                    "department": row["市级部门"],
                    "power_type": row["权力类型"],
                    "power_name": row["权力名称"],
                    "metadata": metadata
                }
            )

    logger.info(f"共读取 {len(all_documents)} 条行政权力清单数据")

    # 批量生成向量并入库
    total_rows = len(all_documents)
    for i in tqdm(range(0, total_rows, BATCH_SIZE), desc="Processing gov powers"):
        batch = all_documents[i : i + BATCH_SIZE]
        texts_to_embed = [doc["text"] for doc in batch]

        vectors = embed_model.encode(texts_to_embed, normalize_embeddings=True)

        data_to_insert = []
        for j, doc in enumerate(batch):
            data_to_insert.append(
                {
                    "vector": vectors[j],
                    "text": doc["text"],
                    "department": doc["department"],
                    "power_type": doc["power_type"],
                    "power_name": doc["power_name"],
                    "doc_type": "gov_power",
                    "metadata": doc["metadata"]
                }
            )

        client.insert(GOV_POWERS_COLLECTION_NAME, data_to_insert)

    logger.info(f"行政权力清单入库完成！共导入 {total_rows} 条数据")


def test_search(
    client,
    embed_model: SentenceTransformer,
    collection_name: str,
    query: str,
    db_type: str = "milvus",
):
    """测试检索"""
    logger.info(f"测试检索 {db_type} '{collection_name}': '{query}'")
    query_vec = embed_model.encode([query], normalize_embeddings=True)

    if db_type == "postgres":
        res = client.search(
            collection_name=collection_name,
            vectors=query_vec.tolist(),
            top_k=3,
            output_fields=["text", "department", "distance"],
        )
        for rank, hit in enumerate(res[0]):
            logger.info(f"Rank {rank+1} (distance: {hit.get('distance', 0):.4f})")
            logger.info(f"部门: {hit.get('department')}")
            logger.info(f"内容摘要: {hit.get('text', '')[:100]}...")
    else:
        res = client.search(
            collection_name=collection_name,
            vectors=query_vec,
            top_k=3,
            output_fields=["text", "department"],
        )

        for rank, hit in enumerate(res[0]):
            logger.info(f"Rank {rank+1} (Distance: {hit['distance']:.4f})")
            logger.info(f"部门: {hit['entity'].get('department')}")
            logger.info(f"内容摘要: {hit['entity'].get('text')[:100]}...")


def main():
    parser = argparse.ArgumentParser(description="数据导入脚本")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["cases", "powers", "all"],
        default="all",
        help="导入模式: cases=问政案例, powers=行政权力清单, all=全部",
    )
    parser.add_argument(
        "--db-type",
        type=str,
        choices=["milvus", "postgres"],
        default=settings.db_type,
        help="数据库类型: milvus 或 postgres",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="强制重建集合/表（删除后重建）",
    )

    args = parser.parse_args()

    # 加载模型
    device = get_device()
    logger.info(f"加载 Embedding 模型: {MODEL_PATH} ...")
    try:
        embed_model = SentenceTransformer(MODEL_PATH, device=device)
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        return

    # 初始化数据库客户端
    db_type = args.db_type
    logger.info(f"使用数据库类型: {db_type}")

    if db_type == "postgres":
        os.makedirs(os.path.dirname(MILVUS_DB_PATH), exist_ok=True)
        client = PostgresDBClient()
    else:
        os.makedirs(os.path.dirname(MILVUS_DB_PATH), exist_ok=True)
        # MilvusDBClient 返回的是单例包装类，需要获取内部 client
        milvus_wrapper = MilvusDBClient()
        client = milvus_wrapper

    # 根据模式执行导入
    if args.mode in ["cases", "all"]:
        logger.info("=" * 50)
        logger.info("开始导入问政案例数据...")
        logger.info("=" * 50)
        ingest_cases(client, embed_model, db_type=db_type)

    if args.mode in ["powers", "all"]:
        logger.info("=" * 50)
        logger.info("开始导入行政权力清单数据...")
        logger.info("=" * 50)
        # powers 模式默认强制重建，因为行政权力清单变更不频繁
        if args.mode == "powers" or args.force:
            ingest_gov_powers(client, embed_model, db_type=db_type)
        else:
            # 检查是否已存在
            if client.has_collection(GOV_POWERS_COLLECTION_NAME):
                logger.info(
                    f"表/集合 {GOV_POWERS_COLLECTION_NAME} 已存在，跳过。如需重新导入请使用 --force"
                )
            else:
                ingest_gov_powers(client, embed_model, db_type=db_type)

    # 测试检索
    if args.mode == "cases":
        test_search(
            client,
            embed_model,
            GOV_CASES_COLLECTION_NAME,
            "雨露计划什么时候发？",
            db_type=db_type,
        )
    elif args.mode == "powers":
        test_search(
            client,
            embed_model,
            GOV_POWERS_COLLECTION_NAME,
            "公积金提取",
            db_type=db_type,
        )


if __name__ == "__main__":
    main()
