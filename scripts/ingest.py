import os
import sys
import pandas as pd
from tqdm import tqdm
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
import torch


# ================= 配置区域 =================
BATCH_SIZE = 16  
MODEL_PATH = "./models/bge-m3"
DB_PATH = "data/milvus_db/gov_pulse.db"
COLLECTION_NAME = "gov_cases"
DATA_PATH = "data/raw/wzlz_municipal_has_reply.xlsx" 

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def init_milvus(client):
    """初始化数据库集合 Schema"""
    if client.has_collection(COLLECTION_NAME):
        print(f"检测到集合 {COLLECTION_NAME} 已存在，正在删除重建...")
        client.drop_collection(COLLECTION_NAME)

    print("🔨 创建新集合 Schema...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=1024, # BGE-M3 维度
        metric_type="COSINE",
        auto_id=True,
        enable_dynamic_field=True 
    )

def process_and_ingest():
    # 1. 读取数据
    if not os.path.exists(DATA_PATH):
        print(f"❌ 错误：找不到文件 {DATA_PATH}，请检查路径！")
        return

    print(f"📖 读取数据: {DATA_PATH}")
    df = pd.read_excel(DATA_PATH)
    
    # === 关键修改：列名映射 ===
    # 将你的 Excel 中文列名映射为代码变量
    # 逻辑：只要包含这些关键字的列，就重命名
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
        print(f"❌ 列名匹配失败！当前列名: {df.columns.tolist()}")
        print("请确保 Excel 包含：'问政内容' 和 '回复内容'")
        return

    # 清洗：去掉没有问题或没有回答的数据
    df = df.dropna(subset=['question', 'answer'])
    # 简单清洗：转为字符串，防止 Excel 里的数字报错
    df['question'] = df['question'].astype(str)
    df['answer'] = df['answer'].astype(str)
    df['department'] = df['department'].astype(str)
        
    print(f"有效数据量: {len(df)} 条")

    # 2. 加载模型
    device = get_device()
    print(f"📥 加载 Embedding 模型: {MODEL_PATH} ...")
    try:
        embed_model = SentenceTransformer(MODEL_PATH, device=device)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 3. 初始化 Milvus
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    client = MilvusClient(DB_PATH)
    init_milvus(client)

    # 4. 批量处理
    total_rows = len(df)
    print("🚀 开始向量化并入库...")
    
    for i in tqdm(range(0, total_rows, BATCH_SIZE), desc="Processing"):
        batch = df.iloc[i : i + BATCH_SIZE]
        
        # === 关键策略：Embedding 谁？===
        # 策略：我们向量化“问题”（A列）。
        # 因为用户的提问通常和 A 列最相似（都是求助、咨询）。
        # 如果我们向量化“回答”，用户问“怎么取钱”，回答是“携带身份证...”，语义匹配度反而可能不高。
        texts_to_embed = batch['question'].tolist()
        
        # 准备其他字段
        answers = batch['answer'].tolist()
        departments = batch['department'].tolist()
        
        # 生成向量
        vectors = embed_model.encode(texts_to_embed, normalize_embeddings=True)
        
        data_to_insert = []
        for j, question_text in enumerate(texts_to_embed):
            # === 关键策略：RAG 上下文存什么？===
            # 我们把“问题”和“回答”拼在一起存入 `text` 字段。
            # 这样检索出来给大模型看的时候，大模型能看到完整的上下文。
            rag_context = f"市民诉求：{question_text}\n官方回复：{answers[j]}"
            
            data_to_insert.append({
                "vector": vectors[j],
                "text": rag_context,            # 给大模型看的内容
                "department": departments[j],   # 过滤用的标签
                "metadata": {                   # 原始数据备份
                    "question": question_text,
                    "answer": answers[j]
                }
            })
            
        client.insert(COLLECTION_NAME, data_to_insert)

    print(f"\n🎉 入库完成！数据库: {DB_PATH}")

    # 5. 验证测试
    test_query = "雨露计划什么时候发？"
    print(f"\n🔎 测试检索: '{test_query}'")
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