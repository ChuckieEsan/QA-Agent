import os
import sys
import pandas as pd
import hashlib
from tqdm import tqdm
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
import torch

sys.path.append(os.getcwd())

from app.core.config import settings
from app.utils import *

# ================= 配置区域 =================
BATCH_SIZE = 16  
MODEL_PATH = str(settings.MODEL_PATHS["embedding"])
DB_PATH = str(settings.MILVUS_DB_PATH)
COLLECTION_NAME = settings.COLLECTION_NAME
DATA_PATH = str(settings.RAW_DATA_PATH)


def init_milvus(client):
    """初始化数据库集合 Schema"""
    if client.has_collection(COLLECTION_NAME):
        print(f"检测到集合 {COLLECTION_NAME} 已存在，正在删除重建...")
        client.drop_collection(COLLECTION_NAME)

    print("创建新集合 Schema...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=1024, # BGE-M3 维度
        metric_type="COSINE",
        auto_id=True,
        enable_dynamic_field=True 
    )

def process_and_ingest():
    print(f"读取数据: {DATA_PATH}")
    df = pd.read_excel(DATA_PATH)
    
    # 列名映射
    rename_map = {}
    for col in df.columns:
        if "问政内容" in col:
            rename_map[col] = "question"
        elif "回复单位" in col:
            rename_map[col] = "department"
        elif "回复内容" in col:
            rename_map[col] = "answer"
            
    df = df.rename(columns=rename_map)
    
    # 检查是否映射成功
    required_cols = ['question', 'answer']
    if not all(col in df.columns for col in required_cols):
        print(f"列名匹配失败！当前列名: {df.columns.tolist()}")
        print("请确保 Excel 包含：'问政内容' 和 '回复内容'")
        return

    # 清洗数据
    df = df.dropna(subset=['question', 'answer'])
    df['question'] = df['question'].astype(str)
    df['answer'] = df['answer'].astype(str)
    df['department'] = df['department'].astype(str)
        
    print(f"有效数据量: {len(df)} 条")

    # 2. 加载模型
    device = get_device()
    print(f"加载 Embedding 模型: {MODEL_PATH} ...")
    try:
        embed_model = SentenceTransformer(MODEL_PATH, device=device)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 3. 初始化 Milvus
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    client = MilvusClient(DB_PATH)
    init_milvus(client)

    # 4. 批量处理
    total_rows = len(df)
    print("开始向量化并入库...")
    
    for i in tqdm(range(0, total_rows, BATCH_SIZE), desc="Processing"):
        batch = df.iloc[i : i + BATCH_SIZE]
        
        texts_to_embed = batch['question'].tolist()
        answers = batch['answer'].tolist()
        departments = batch['department'].tolist()
        
        # 生成向量
        vectors = embed_model.encode(texts_to_embed, normalize_embeddings=True)
        
        data_to_insert = []
        for j, question_text in enumerate(texts_to_embed):
            # === 生成唯一 ID ===
            # 直接在这里生成 doc_id，替代后续的迁移脚本
            current_answer = answers[j]
            doc_id = generate_doc_id(question_text, current_answer)

            # RAG 上下文
            rag_context = f"市民诉求：{question_text}\n官方回复：{current_answer}"
            
            data_to_insert.append({
                "vector": vectors[j],
                "text": rag_context,            
                "department": departments[j],   
                "metadata": {                   
                    "doc_id": doc_id,
                    "question": question_text,
                    "answer": current_answer
                }
            })
            
        client.insert(COLLECTION_NAME, data_to_insert)

    print(f"\n入库完成！数据库: {DB_PATH}")

    # 5. 验证测试
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