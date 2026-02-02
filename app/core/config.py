import os
from pathlib import Path
from typing import Dict, Optional
from pydantic import BaseModel, Field, ConfigDict
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


class BaseConfig(BaseModel):
    """所有配置的基类，解决命名冲突"""
    model_config = ConfigDict(
        protected_namespaces=(),  # 禁用命名空间保护
        extra='ignore'  # 可选：忽略额外字段
    )

class PathConfig(BaseConfig):
    """路径配置"""

    project_root: Path = Field(default=PROJECT_ROOT, description="项目根目录")
    data_dir: Path = Field(default=PROJECT_ROOT / "data", description="数据目录")
    model_dir: Path = Field(default=PROJECT_ROOT / "models", description="模型目录")
    log_dir: Path = Field(default=PROJECT_ROOT / "logs", description="日志目录")
    cache_dir: Path = Field(default=PROJECT_ROOT / ".cache", description="缓存目录")

    # 原始数据路径
    raw_data_db_path: Path = Field(
        default=PROJECT_ROOT / "data" / "raw_data.db",
        description="原始数据SQLite数据库路径",
    )

    # 处理数据路径
    processed_data_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "processed", description="处理后的数据目录"
    )
    query_test_data_path: Path = Field(
        default=PROJECT_ROOT / "data" / "processed" / "query_test_data.jsonl",
        description="查询测试数据路径",
    )

    # 向量数据库路径
    milvus_db_path: str = Field(
        default=str(PROJECT_ROOT / "data" / "milvus_db" / "gov_pulse.db"),
        description="Milvus数据库路径",
    )


class ModelConfig(BaseConfig):
    """模型配置"""

    # Embedding模型
    embedding_model: str = Field(default="bge-m3", description="Embedding模型名称")
    embedding_model_path: Path = Field(
        default=PROJECT_ROOT / "models" / "bge-m3", description="Embedding模型本地路径"
    )
    embedding_size: int = Field(
        default=1024, description="Embedding向量维度（BGE-M3为1024）"
    )

    # 重排模型
    reranker_model: Optional[str] = Field(
        default=None, description="重排模型名称（可选）"
    )
    reranker_model_path: Optional[Path] = Field(
        default=None, description="重排模型本地路径（可选）"
    )


class MilvusDBConfig(BaseConfig):
    """向量数据库配置"""

    collection_name: str = Field(default="gov_cases", description="集合名称")
    vector_dimension: int = Field(default=1024, description="向量维度（BGE-M3为1024）")
    metric_type: str = Field(default="COSINE", description="相似度度量类型")
    enable_dynamic_field: bool = Field(default=True, description="是否启用动态字段")

    # 检索参数
    default_top_k: int = Field(default=5, description="默认返回结果数量")
    max_top_k: int = Field(default=20, description="最大返回结果数量")

    # 性能优化
    search_cache_size: int = Field(default=100, description="搜索缓存大小")
    search_cache_ttl: int = Field(default=300, description="搜索缓存过期时间（秒）")


class LLMConfig(BaseConfig):
    """大语言模型配置"""

    provider: str = Field(
        default="qwen", description="LLM提供商 (qwen/openai/azure/ollama)"
    )
    model_name: str = Field(default="qwen-max", description="模型名称")
    api_key: str = Field(default="", description="API密钥")
    api_base: str = Field(default="https://dashscope.aliyuncs.com/compatible-mode/v1", description="API基础URL")

    # 生成参数
    temperature: float = Field(default=0.1, description="温度参数", ge=0.0, le=2.0)
    max_tokens: int = Field(default=2000, description="最大生成token数")
    top_p: float = Field(default=0.9, description="Top-p采样参数")

    # 上下文配置
    max_context_length: int = Field(default=4000, description="最大上下文长度")
    enable_streaming: bool = Field(default=False, description="是否启用流式输出")


class LoggingConfig(BaseConfig):
    """日志配置"""

    level: str = Field(
        default="INFO", description="日志级别 (DEBUG/INFO/WARNING/ERROR)"
    )
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="日志格式",
    )
    file_enabled: bool = Field(default=True, description="是否启用文件日志")
    file_path: Path = Field(
        default=PROJECT_ROOT / "logs" / "govpulse.log", description="日志文件路径"
    )
    max_file_size: int = Field(
        default=10 * 1024 * 1024, description="最大日志文件大小（字节）"  # 10MB
    )
    backup_count: int = Field(default=5, description="备份文件数量")


class PerformanceConfig(BaseConfig):
    """性能配置"""

    # 监控配置
    enable_metrics: bool = Field(default=True, description="是否启用性能监控")
    metrics_port: int = Field(default=9090, description="监控指标端口")

    # 性能优化
    batch_size: int = Field(default=16, description="批量处理大小")
    max_workers: int = Field(default=4, description="最大工作线程数")
    enable_profiling: bool = Field(default=False, description="是否启用性能分析")

    # 超时配置
    request_timeout: int = Field(default=30, description="请求超时时间（秒）")
    connect_timeout: int = Field(default=10, description="连接超时时间（秒）")


class RetrieverConfig(BaseConfig):
    """检索器配置"""

    # 阈值策略
    threshold_strategy: str = Field(
        default="hybrid", description="阈值策略 (hybrid/fixed/dynamic/top_percentage)"
    )
    base_threshold: float = Field(
        default=0.65, description="基础相似度阈值", ge=0.0, le=1.0
    )
    min_results: int = Field(default=3, description="最小返回结果数", ge=1)
    max_results: int = Field(default=10, description="最大返回结果数", ge=1)

    # 重排权重配置
    weight_similarity: float = Field(
        default=0.8, description="相似度权重", ge=0.0, le=1.0
    )
    weight_recency: float = Field(default=0.7, description="时效性权重", ge=0.0, le=1.0)
    weight_authority: float = Field(
        default=0.2, description="部门权威性权重", ge=0.0, le=1.0
    )
    weight_length: float = Field(
        default=0.1, description="内容长度权重", ge=0.0, le=1.0
    )

    # 部门权威性映射
    department_authority: Dict[str, float] = Field(
        default={"default": 0.5}, description="部门权威性分数映射"
    )

    # 时间衰减配置
    recency_weights: Dict[str, float] = Field(
        default={
            "within_week": 1.0,
            "within_month": 0.9,
            "within_quarter": 0.7,
            "within_year": 0.5,
            "beyond_year": 0.3,
        },
        description="时间衰减权重",
    )

    # 缓存配置
    enable_cache: bool = Field(default=True, description="是否启用缓存")
    cache_max_size: int = Field(default=100, description="缓存最大条目数")
    cache_ttl_minutes: int = Field(default=5, description="缓存过期时间（分钟）")


class Settings(BaseConfig):
    """
    主配置类，聚合所有子配置
    """
    # 基础信息
    project_name: str = Field(default="GovPulse", description="项目名称")
    version: str = Field(default="1.0.0", description="版本号")
    debug: bool = Field(default=False, description="调试模式")
    
    # 子配置
    paths: PathConfig = Field(default_factory=PathConfig, description="路径配置")
    models: ModelConfig = Field(default_factory=ModelConfig, description="模型配置")
    vectordb: MilvusDBConfig = Field(default_factory=MilvusDBConfig, description="向量数据库配置")
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig, description="检索器配置")
    llm: LLMConfig = Field(default_factory=LLMConfig, description="LLM配置")
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="日志配置")
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig, description="性能配置")
    
    # 动态配置（从环境变量加载）
    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
    
    def __init__(self, **kwargs):
        # 在初始化前处理环境变量覆盖
        super().__init__(**kwargs)
        
        # 设置API密钥（从环境变量）
        if not self.llm.api_key:
            self.llm.api_key = os.getenv("DASHSCOPE_API_KEY", "")
        
        # 自动创建必要目录
        self._create_directories()
    
    def _create_directories(self):
        """创建必要的目录结构"""
        dirs_to_create = [
            self.paths.data_dir,
            self.paths.model_dir,
            self.paths.log_dir,
            self.paths.cache_dir,
            self.paths.processed_data_dir,
            self.logging.file_path.parent
        ]
        
        for directory in dirs_to_create:
            if isinstance(directory, Path):
                directory.mkdir(parents=True, exist_ok=True)


# 实例化配置对象 (单例模式)
settings = Settings()

# 调试代码：直接运行 python app/core/config.py 可以检查路径对不对
if __name__ == "__main__":
    print("=" * 60)
    print(f"项目名称: {settings.project_name} v{settings.version}")
    print("=" * 60)
    
    # 显示关键路径
    print("\n📁 关键路径:")
    print(f"  项目根目录: {settings.paths.project_root}")
    print(f"  数据目录: {settings.paths.data_dir}")
    print(f"  模型目录: {settings.paths.model_dir}")
    print(f"  日志目录: {settings.paths.log_dir}")
    
    # 显示模型配置
    print("\n🤖 模型配置:")
    print(f"  Embedding模型: {settings.models.embedding_model}")
    print(f"  模型路径: {settings.models.embedding_model_path}")
    print(f"  LLM提供商: {settings.llm.provider}")
    print(f"  LLM模型: {settings.llm.model_name}")
    
    # 显示向量数据库配置
    print("\n🗄️ 向量数据库配置:")
    print(f"  集合名称: {settings.vectordb.collection_name}")
    print(f"  向量维度: {settings.vectordb.vector_dimension}")
    print(f"  数据库路径: {settings.paths.milvus_db_path}")
    
    # 显示检索器配置
    print("\n🔍 检索器配置:")
    print(f"  阈值策略: {settings.retriever.threshold_strategy}")
    print(f"  基础阈值: {settings.retriever.base_threshold}")
    print(f"  重排权重: S={settings.retriever.weight_similarity}, "
          f"R={settings.retriever.weight_recency}, "
          f"A={settings.retriever.weight_authority}, "
          f"L={settings.retriever.weight_length}")