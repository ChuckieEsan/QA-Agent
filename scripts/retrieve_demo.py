import os
import sys
import time
import torch
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

sys.path.append(os.getcwd())

# ================= 配置加载逻辑 =================
try:
    from app.core.config import settings
    MODEL_PATH = str(settings.MODEL_PATHS['embedding'])
    DB_PATH = settings.MILVUS_DB_PATH
    COLLECTION_NAME = settings.COLLECTION_NAME
    print(f"✅ 从 Config 加载配置")
except ImportError:
    print("⚠️ 未找到 Config，使用默认硬编码路径")
    MODEL_PATH = "./models/bge-m3"
    DB_PATH = "data/milvus_db/gov_pulse.db"
    COLLECTION_NAME = "gov_cases"

# 默认检索数量
DEFAULT_TOP_K = 5

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def load_resources():
    """只加载一次模型和数据库连接"""
    print("-" * 50)
    print("正在初始化系统资源，请稍候...")
    
    # 1. 检查数据库是否存在
    if not os.path.exists(os.path.dirname(DB_PATH)):
        print(f"❌ 数据库文件不存在: {DB_PATH}")
        print("请先运行 ingest.py 进行数据入库！")
        sys.exit(1)

    # 2. 连接 Milvus
    try:
        client = MilvusClient(DB_PATH)
        print(f"✅ 数据库连接成功: {DB_PATH}")
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")
        sys.exit(1)

    # 3. 加载模型
    device = get_device()
    print(f"📥 正在加载 Embedding 模型 ({device}): {MODEL_PATH} ...")
    try:
        model = SentenceTransformer(MODEL_PATH, device=device)
        print("✅ 模型加载完成")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        sys.exit(1)

    return client, model

def search(client, model, query, top_k=DEFAULT_TOP_K):
    """执行单次检索"""
    start_time = time.time()
    
    # 1. 向量化
    query_vec = model.encode([query], normalize_embeddings=True)
    
    # 2. 检索
    res = client.search(
        collection_name=COLLECTION_NAME,
        data=query_vec,
        limit=top_k,
        output_fields=["text", "department", "metadata"] # 获取字段
    )
    
    end_time = time.time()
    latency = (end_time - start_time) * 1000 # 转毫秒
    
    return res[0], latency

def main():
    # 初始化
    client, model = load_resources()
    
    print("-" * 50)
    print(f"🚀 交互式检索系统已就绪！(Top-K = {DEFAULT_TOP_K})")
    print("💡 输入 'exit' 或 'quit' 退出")
    print("-" * 50)

    while True:
        try:
            # 获取输入，使用带颜色的提示符（如果终端支持）
            query = input("\n🙋 请输入市民诉求: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['exit', 'quit']:
                print("👋 再见！")
                break
                
            # 执行检索
            print(f"🔍 正在检索...")
            hits, latency = search(client, model, query)
            
            # 打印结果
            print(f"\n✅ 检索完成 | 耗时: {latency:.2f}ms | 命中: {len(hits)} 条")
            print("=" * 60)
            
            for rank, hit in enumerate(hits):
                score = hit['distance']
                dept = hit['entity'].get('department', '未知部门')
                content = hit['entity'].get('text', '')
                
                # 提取纯问题部分用于展示（如果metadata里存了的话）
                # 也可以直接展示完整的 text
                
                # 颜色区分（高分绿色，低分红色，需终端支持，这里用简单符号代替）
                score_icon = "⭐" if score > 0.6 else "  "
                
                print(f"Rank {rank+1} [{score:.4f}] {score_icon} | 🏛️ {dept}")
                print(f"📄 内容摘要: {content[:150]}...") # 只显示前150字
                print("-" * 60)
                
        except KeyboardInterrupt:
            print("\n👋 用户中断，退出系统。")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")

if __name__ == "__main__":
    main()