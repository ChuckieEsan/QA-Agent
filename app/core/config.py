import os
from pathlib import Path
from dotenv import load_dotenv

# __file__ 是 config.py 的位置
# .parent -> app/core/
# .parent.parent -> app/
# .parent.parent.parent -> GovPulse/ (项目根目录)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    load_dotenv(PROJECT_ROOT / ".env")
    print(f"已加载环境变量: {env_path}")
else:
    print(f"未找到环境变量文件: {env_path}，使用系统环境变量")

class Settings:
    # 基础信息
    PROJECT_NAME: str = "GovPulse"
    VERSION: str = "1.0.0"

    # ================= 路径配置 =================
    DATA_DIR: Path = PROJECT_ROOT / "data"
    MODEL_DIR: Path = PROJECT_ROOT / "models"
    
    # 原始数据路径
    RAW_DATA_PATH: Path = DATA_DIR / "raw" / "wzlz_municipal_has_reply.xlsx"

    QUERY_TEST_DATA_PATH: Path = DATA_DIR / "processed" / "query_test_data.jsonl"

    # ================= 模型配置 =================
    # 这里的 Key 是你在代码里调用的名字，Value 是实际路径
    MODEL_PATHS: dict = {
        "embedding": MODEL_DIR / "bge-m3",
        # 如果你未来加 Rerank，直接在这里加一行即可
        # "reranker": MODEL_DIR / "bge-reranker-base",
    }

    # ================= 向量库配置 =================
    MILVUS_DB_PATH: str = str(DATA_DIR / "milvus_db" / "gov_pulse.db")
    COLLECTION_NAME: str = "gov_cases"
    VECTOR_DIM: int = 1024
    BATCH_SIZE: int = 16

    # ================= 其他配置 =================
    QWEN_API_KEY: str = os.getenv("DASHSCOPE_API_KEY", "")

# 实例化配置对象 (单例模式)
settings = Settings()

# 调试代码：直接运行 python app/core/config.py 可以检查路径对不对
if __name__ == "__main__":
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"Excel路径: {settings.RAW_DATA_PATH}")
    print(f"模型路径: {settings.MODEL_PATHS['embedding']}")
    print(f"API Key: {settings.QWEN_API_KEY[:6]}****")