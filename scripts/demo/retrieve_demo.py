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
COLLECTION_NAME = settings.vectordb.collection_name
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
    
    logger.info("-" * 50)
    logger.info(f"交互式检索系统已就绪！(Top-K = {DEFAULT_TOP_K})")
    logger.info("输入 'exit' 或 'quit' 退出")
    logger.info("-" * 50)

    while True:
        try:
            # 获取输入，使用带颜色的提示符（如果终端支持）
            query = input("\n🙋 请输入市民诉求: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['exit', 'quit']:
                logger.info("再见！")
                break
                
            # 执行检索
            logger.info("正在检索...")
            hits, latency = search(client, model, query)
            
            # 打印结果
            logger.info(f"检索完成 | 耗时: {latency:.2f}ms | 命中: {len(hits)} 条")
            logger.info("=" * 60)
            
            for rank, hit in enumerate(hits):
                score = hit['distance']
                dept = hit['entity'].get('department', '未知部门')
                content = hit['entity'].get('text', '')
                
                # 提取纯问题部分用于展示（如果metadata里存了的话）
                # 也可以直接展示完整的 text
                
                # 颜色区分（高分绿色，低分红色，需终端支持，这里用简单符号代替）
                score_icon = "⭐" if score > 0.6 else "  "
                
                logger.debug(f"Rank {rank+1} [{score:.4f}] {score_icon} | {dept}")
                logger.debug(f"内容摘要: {content[:150]}...") # 只显示前150字
                logger.debug("-" * 60)
                
        except KeyboardInterrupt:
            logger.info("用户中断，退出系统。")
            break
        except Exception as e:
            logger.error(f"发生错误: {e}")

if __name__ == "__main__":
    main()