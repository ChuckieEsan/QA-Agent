import os
import sys
import time
import torch
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

sys.path.append(os.getcwd())

from src.config.setting import settings
from src.app.infra.utils.logger import get_logger

logger = get_logger(__name__)


MODEL_PATH = str(settings.models.embedding_model_path)
DB_PATH = str(settings.vectordb.db_path)
GOV_CASES_COLLECTION_NAME = settings.vectordb.gov_cases_collection_name
GOV_POWERS_COLLECTION_NAME = settings.vectordb.gov_powers_collection_name
logger.info("从 Config 加载配置")

# 默认检索数量
DEFAULT_TOP_K = 5

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def load_resources():
    """只加载一次模型和数据库连接"""
    logger.info("-" * 50)
    logger.info("正在初始化系统资源，请稍候...")
    
    # 1. 检查数据库是否存在
    if not os.path.exists(os.path.dirname(DB_PATH)):
        logger.error(f"数据库文件不存在: {DB_PATH}")
        logger.error("请先运行 ingest.py 进行数据入库！")
        sys.exit(1)

    # 2. 连接 Milvus
    try:
        client = MilvusClient(DB_PATH)
        logger.info(f"数据库连接成功: {DB_PATH}")
    except Exception as e:
        logger.error(f"数据库连接失败: {e}")
        sys.exit(1)

    # 3. 加载模型
    device = get_device()
    logger.info(f"正在加载 Embedding 模型 ({device}): {MODEL_PATH} ...")
    try:
        model = SentenceTransformer(MODEL_PATH, device=device)
        logger.info("模型加载完成")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        sys.exit(1)

    return client, model

def search(client, model, collection_name, query, top_k=DEFAULT_TOP_K):
    """执行单次检索"""
    start_time = time.time()

    # 1. 向量化
    query_vec = model.encode([query], normalize_embeddings=True)

    # 2. 检索
    res = client.search(
        collection_name=collection_name,
        data=query_vec,
        limit=top_k,
        output_fields=["text", "department", "metadata"]  # 获取字段
    )

    end_time = time.time()
    latency = (end_time - start_time) * 1000  # 转毫秒

    return res[0], latency


def select_collection(client):
    """让用户选择要检索的 collection"""
    available_collections = []

    # 检查哪些 collection 存在
    for name in [GOV_CASES_COLLECTION_NAME, GOV_POWERS_COLLECTION_NAME]:
        if client.has_collection(name):
            available_collections.append(name)

    if not available_collections:
        logger.error("没有可用的 collection，请先运行 ingest.py 进行数据入库！")
        return None

    logger.info("\n请选择要检索的 collection:")
    for i, name in enumerate(available_collections):
        if name == GOV_CASES_COLLECTION_NAME:
            desc = "问政案例"
        elif name == GOV_POWERS_COLLECTION_NAME:
            desc = "行政权力清单"
        else:
            desc = "未知"
        logger.info(f"  [{i + 1}] {name} ({desc})")

    while True:
        try:
            choice = input(f"\n请输入序号 (1-{len(available_collections)}) 或按回车使用默认: ").strip()
            if not choice:
                # 默认选择第一个
                return available_collections[0]
            idx = int(choice) - 1
            if 0 <= idx < len(available_collections):
                return available_collections[idx]
            else:
                logger.warning(f"请输入 1-{len(available_collections)} 之间的数字")
        except ValueError:
            logger.warning("请输入有效的数字")


def main():
    # 初始化
    client, model = load_resources()

    # 选择 collection
    collection_name = select_collection(client)
    if not collection_name:
        sys.exit(1)

    logger.info("-" * 50)
    logger.info(f"当前检索 collection: {collection_name}")
    logger.info(f"交互式检索系统已就绪！(Top-K = {DEFAULT_TOP_K})")
    logger.info("输入 'exit' 或 'quit' 退出")
    logger.info("输入 'switch' 切换 collection")
    logger.info("-" * 50)

    while True:
        try:
            query = input("\n🙋 请输入查询内容: ").strip()

            if not query:
                continue

            if query.lower() in ['exit', 'quit']:
                logger.info("再见！")
                break

            if query.lower() == 'switch':
                collection_name = select_collection(client)
                if not collection_name:
                    sys.exit(1)
                logger.info(f"已切换到 collection: {collection_name}")
                continue

            # 执行检索
            logger.info("正在检索...")
            hits, latency = search(client, model, collection_name, query)

            # 打印结果
            logger.info(f"检索完成 | 耗时: {latency:.2f}ms | 命中: {len(hits)} 条")
            logger.info("=" * 60)

            for rank, hit in enumerate(hits):
                score = hit['distance']
                dept = hit['entity'].get('department', '未知部门')
                content = hit['entity'].get('text', '')

                score_icon = "⭐" if score > 0.6 else "  "

                logger.info(f"Rank {rank+1} [{score:.4f}] {score_icon} | {dept}")
                logger.info(f"内容摘要: {content[:150]}...")  # 只显示前150字
                logger.info("-" * 60)

        except KeyboardInterrupt:
            logger.info("用户中断，退出系统。")
            break
        except Exception as e:
            logger.error(f"发生错误: {e}")

if __name__ == "__main__":
    main()