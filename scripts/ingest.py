import os
import sys
import sqlite3
from tqdm import tqdm
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

sys.path.append(os.getcwd())

from app.core.config import settings
from app.utils import generate_doc_id, get_device, clean_text

# ================= 配置区域 =================
BATCH_SIZE = 16  
MODEL_PATH = str(settings.models.embedding_model_path)
MILVUS_DB_PATH = str(settings.paths.milvus_db_path)
COLLECTION_NAME = settings.vectordb.collection_name
SQLITE_DB_PATH = str(settings.paths.raw_data_db_path)

def init_milvus(client: MilvusClient):
    """初始化数据库集合 Schema"""
    if client.has_collection(COLLECTION_NAME):
        print(f"检测到集合 {COLLECTION_NAME} 已存在，正在删除重建...")
        client.drop_collection(COLLECTION_NAME)

    print("🔨 创建新集合 Schema...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=settings.models.embedding_size, # BGE-M3 维度
        metric_type="COSINE",
        auto_id=True,
        enable_dynamic_field=True 
    )

def fetch_data_from_sqlite(db_path: str):
    """从 SQLite 读取所有已爬取的数据"""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"找不到数据库文件: {db_path}")
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row # 允许通过列名访问
    cursor = conn.cursor()
    
    # 仅读取有标题、问题和回复的数据
    cursor.execute("""
        SELECT id, title, dept, question, answer, question_time, url 
        FROM wenzheng 
        WHERE question IS NOT NULL AND answer IS NOT NULL
    """)
    rows = cursor.fetchall()
    conn.close()
    return rows

def process_and_ingest():
    # 1. 读取数据
    try:
        rows = fetch_data_from_sqlite(SQLITE_DB_PATH)
    except Exception as e:
        print(f"❌ 读取 SQLite 失败: {e}")
        return

    print(f"📖 有效数据量: {len(rows)} 条")
    if len(rows) == 0:
        print("⚠️ 数据库为空，请先运行 crawl.py")
        return

    # 2. 加载模型
    device = get_device()
    print(f"📥 加载 Embedding 模型: {MODEL_PATH} ...")
    try:
        embed_model = SentenceTransformer(MODEL_PATH, device=device)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 3. 初始化 Milvus
    os.makedirs(os.path.dirname(MILVUS_DB_PATH), exist_ok=True)
    client = MilvusClient(MILVUS_DB_PATH)
    init_milvus(client)

    # 4. 批量处理
    total_rows = len(rows)
    print("🚀 开始向量化并入库...")
    
    # 将 sqlite3.Row 对象转换为字典列表，方便处理
    data_list = [dict(row) for row in rows]
    
    for i in tqdm(range(0, total_rows, BATCH_SIZE), desc="Processing"):
        batch = data_list[i : i + BATCH_SIZE]
        
        # 准备 Embedding 的文本：主要使用“问题”
        texts_to_embed = [clean_text(item['question']) for item in batch]
        
        # 生成向量
        vectors = embed_model.encode(texts_to_embed, normalize_embeddings=True)
        
        data_to_insert = []
        for j, item in enumerate(batch):
            # 清洗文本
            question_text = clean_text(item['question'])
            answer_text = clean_text(item['answer'])
            title_text = item['title']
            dept_text = item['dept']
            time_text = item['question_time']
            url_text = item['url']
            
            doc_id = generate_doc_id(question_text, answer_text)

            # === 构建 RAG 上下文 (Rich Context) ===
            # 这里加入了标题、时间、来源，让大模型回答时更专业
            rag_context = (
                f"标题：{title_text}\n"
                f"部门：{dept_text}\n"
                f"时间：{time_text}\n"
                f"市民诉求：{question_text}\n"
                f"官方回复：{answer_text}\n"
                f"来源链接：{url_text}"
            )
            
            data_to_insert.append({
                "vector": vectors[j],
                "text": rag_context,            
                "department": dept_text,   
                "metadata": {                   
                    "doc_id": doc_id,
                    "crawler_id": item['id'], # 原始ID
                    "title": title_text,
                    "question": question_text,
                    "answer": answer_text,
                    "url": url_text,
                    "time": time_text
                }
            })
            
        client.insert(COLLECTION_NAME, data_to_insert)

    print(f"\n🎉 入库完成！数据库: {MILVUS_DB_PATH}")

    # 验证测试
    test_query = "雨露计划什么时候发？"
    print(f"\n测试检索: '{test_query}'")
    query_vec = embed_model.encode([test_query], normalize_embeddings=True)
    
    res = client.search(
        collection_name=COLLECTION_NAME,
        data=query_vec,
        limit=2,
        output_fields=["text", "department"]
    )
    
    for rank, hit in enumerate(res[0]):
        print(f"\n--- Rank {rank+1} (Score: {hit['distance']:.4f}) ---")
        print(f"部门: {hit['entity'].get('department')}")
        print(f"内容摘要: {hit['entity'].get('text')[:100]}...")

if __name__ == "__main__":
    process_and_ingest()