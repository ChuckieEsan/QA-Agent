import pandas as pd
import json
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
import os
import sys

sys.path.append(os.getcwd())

from config.setting import settings

def main():
    # 1. 加载资源
    print(f"正在连接数据库: {settings.MILVUS_DB_PATH}")
    client = MilvusClient(str(settings.MILVUS_DB_PATH))
    
    print(f"正在加载模型: {settings.MODEL_PATHS['embedding']}")
    model = SentenceTransformer(str(settings.MODEL_PATHS["embedding"]), device='cuda') 

    # 2. 加载测试集
    test_data = []
    test_data_path = str(settings.QUERY_TEST_DATA_PATH)
    if not os.path.exists(test_data_path):
        print(f"错误: 找不到测试集文件 {test_data_path}")
        return

    with open(test_data_path, "r", encoding='utf-8') as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line))
    
    total = len(test_data)
    print(f"已加载测试样本: {total} 条")

    # 3. 批量处理 (Batch Processing)
    # 提取所有查询和对应的真值 ID
    queries = [item['test_query'] for item in test_data]
    ground_truth_ids = [item['doc_id'] for item in test_data]

    print("正在批量编码查询向量...")
    query_vectors = model.encode(queries, normalize_embeddings=True, batch_size=32, show_progress_bar=True)

    print("正在执行批量检索...")
    res = client.search(
        collection_name=settings.COLLECTION_NAME,
        data=query_vectors,
        limit=10, 
        output_fields=["metadata"] 
    )

    # 4. 计算指标
    hits_at_5 = 0
    hits_at_10 = 0

    # 遍历每一个查询的检索结果
    for i, hits in enumerate(res):
        target_id = ground_truth_ids[i]
        
        retrieved_ids = []
        for hit in hits:
            # 兼容性处理：防止 metadata 为空
            meta = hit['entity'].get('metadata', {})
            r_id = meta.get('doc_id')
            retrieved_ids.append(r_id)
        
        if target_id in retrieved_ids[:5]:
            hits_at_5 += 1
        if target_id in retrieved_ids[:10]:
            hits_at_10 += 1

        if target_id not in retrieved_ids[:10]:
            print(f"❌ Missed Case:")
            # print(f"Query: {query}")
            # print(f"Truth ID: {ground_truth_id}")
            # 打印排在前面的几个错误答案
            print(f"Top-1 Wrong Result: {res[i][0]['entity']['text'][:50]}...")


    # 5. 输出结果
    print("\n" + "="*30)
    print(f"基线评估结果 (Total: {total})")
    print("="*30)
    print(f"Recall@5 : {hits_at_5/total:.2%}")
    print(f"Recall@10: {hits_at_10/total:.2%}")
    print("="*30)

if __name__ == "__main__":
    main()