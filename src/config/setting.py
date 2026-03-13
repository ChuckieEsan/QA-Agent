import os
from pathlib import Path
from typing import Dict, Optional
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv

# __file__ 是 setting.py 的位置
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    load_dotenv(PROJECT_ROOT / ".env")
    print(f"已加载环境变量：{env_path}")
else:
    print(f"未找到环境变量文件：{env_path}，使用系统环境变量")


class BaseConfig(BaseModel):
    """所有配置的基类，解决命名冲突"""

    model_config = ConfigDict(
        protected_namespaces=(),  # 禁用命名空间保护
        extra="ignore",  # 可选：忽略额外字段
    )


class PathConfig(BaseConfig):
    """路径配置"""

    project_root: Path = Field(default=PROJECT_ROOT, description="项目根目录")
    data_dir: Path = Field(default=PROJECT_ROOT / "data", description="数据目录")
    model_dir: Path = Field(default=PROJECT_ROOT / "models", description="模型目录")
    log_dir: Path = Field(default=PROJECT_ROOT / "logs", description="日志目录")

    # 原始数据路径
    raw_data_db_path: Path = Field(
        default=PROJECT_ROOT / "data" / "raw_data.db",
        description="原始数据 SQLite 数据库路径",
    )

    # 处理数据路径
    processed_data_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "processed", description="处理后的数据目录"
    )
    query_test_data_path: Path = Field(
        default=PROJECT_ROOT / "data" / "processed" / "query_test_data.jsonl",
        description="查询测试数据路径",
    )


class ModelConfig(BaseConfig):
    """模型配置"""

    # Embedding 模型
    embedding_model: str = Field(default="bge-m3", description="Embedding 模型名称")
    embedding_model_path: Path = Field(
        default=PROJECT_ROOT / "models" / "bge-m3", description="Embedding 模型本地路径"
    )
    embedding_size: int = Field(
        default=1024, description="Embedding 向量维度（BGE-M3 为 1024）"
    )

    # 重排模型
    reranker_model: Optional[str] = Field(
        default="bge-reranker-base", description="重排模型名称"
    )
    reranker_model_path: Optional[Path] = Field(
        default=PROJECT_ROOT / "models" / "bge-reranker-base",
        description="重排模型本地路径",
    )


class MilvusDBConfig(BaseConfig):
    """向量数据库配置"""

    db_path: str = Field(
        default=str(PROJECT_ROOT / "data" / "milvus_db" / "gov_pulse.db"),
        description="Milvus 数据库路径",
    )

    gov_cases_collection_name: str = Field(default="gov_cases", description="问政案例集合名称")
    gov_powers_collection_name: str = Field(default="gov_powers", description="行政权力清单集合名称")
    vector_dimension: int = Field(
        default=1024, description="向量维度（BGE-M3 为 1024）"
    )
    metric_type: str = Field(default="COSINE", description="相似度度量类型")
    enable_dynamic_field: bool = Field(default=True, description="是否启用动态字段")

    # 检索参数
    default_top_k: int = Field(default=5, description="默认返回结果数量")
    max_top_k: int = Field(default=20, description="最大返回结果数量")

    # 性能优化
    search_cache_size: int = Field(default=100, description="搜索缓存大小")
    search_cache_ttl: int = Field(default=300, description="搜索缓存过期时间（秒）")


class LLMProviderConfig(BaseConfig):
    """单个模型提供商的配置"""

    provider_id: str = Field(
        default="", description="提供商标识 (deepseek/qwen/ollama)"
    )
    api_key: str = Field(default="", description="API 密钥")
    base_url: str = Field(default="", description="API 基础 URL")
    models: Dict[str, str] = Field(
        default_factory=dict,
        description="模型映射 {用途：模型名称}, 如 {'generation': 'deepseek-reasoner', 'classification': 'deepseek-chat'}",
    )


class LLMConfig(BaseConfig):
    """大语言模型配置

    支持多提供商架构：
    - default_provider: 默认提供商 ID
    - providers: 所有提供商配置字典
    - 向后兼容：保留 heavy/light/optimizer_model 配置用于过渡
    """

    # 多提供商配置
    default_provider: str = Field(default="deepseek", description="默认提供商 ID")
    providers: Dict[str, LLMProviderConfig] = Field(
        default_factory=dict, description="所有提供商配置字典"
    )

    # 上下文配置
    max_context_length: int = Field(default=4000, description="最大上下文长度")
    enable_streaming: bool = Field(default=False, description="是否启用流式输出")

    # LLM 调用次数限制
    max_llm_calls: int = Field(
        default=20, description="每轮对话最大 LLM 调用次数（防无限循环）"
    )

    # 工具调用配置
    max_function_call_retries: int = Field(
        default=2, description="工具调用最大尝试次数"
    )

    def get_provider_config(
        self, provider_id: Optional[str] = None
    ) -> LLMProviderConfig:
        """获取指定提供商配置，如果没有则返回默认提供商配置"""
        pid = provider_id or self.default_provider
        return self.providers[pid]

    def get_model_for_purpose(
        self, purpose: str, provider_id: Optional[str] = None
    ) -> str:
        """获取指定用途的模型名称"""
        provider = self.get_provider_config(provider_id)
        return provider.models.get(purpose, self.heavy_model_name)


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
    min_similarity: float = Field(
        default=0.65, description="最小相似度阈值", ge=0.0, le=1.0
    )
    min_results: int = Field(default=3, description="最小返回结果数", ge=1)
    max_results: int = Field(default=10, description="最大返回结果数", ge=1)

    # 重排权重配置（移除部门权威性，所有部门信息平等对待）
    # 权重说明：
    # - 相似度 (60%): 向量相似度是最重要的指标
    # - 时效性 (30%): 优先展示近期政策和回复
    # - 内容长度 (10%): 适度偏好内容更丰富的案例
    weight_similarity: float = Field(
        default=0.6, description="相似度权重", ge=0.0, le=1.0
    )
    weight_recency: float = Field(default=0.3, description="时效性权重", ge=0.0, le=1.0)
    weight_length: float = Field(
        default=0.1, description="内容长度权重", ge=0.0, le=1.0
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
    vectordb: MilvusDBConfig = Field(
        default_factory=MilvusDBConfig, description="向量数据库配置"
    )
    retriever: RetrieverConfig = Field(
        default_factory=RetrieverConfig, description="检索器配置"
    )
    llm: LLMConfig = Field(default_factory=LLMConfig, description="LLM 配置")
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig, description="日志配置"
    )
    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig, description="性能配置"
    )

    # 从环境变量加载配置
    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"

    def __init__(self, **kwargs):
        # 在初始化前处理环境变量覆盖
        super().__init__(**kwargs)

        # 从环境变量加载多提供商配置
        self._load_provider_configs_from_env()

        # 自动创建必要目录
        self._create_directories()

    def _load_provider_configs_from_env(self):
        """从环境变量加载各提供商配置"""
        providers = {}

        # DeepSeek 配置
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "")
        deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        if deepseek_api_key:
            providers["deepseek"] = LLMProviderConfig(
                provider_id="deepseek",
                api_key=deepseek_api_key,
                base_url=deepseek_base_url,
                models={
                    "generation": os.getenv(
                        "DEEPSEEK_GENERATION_MODEL", "deepseek-chat"
                    ),
                    "classification": os.getenv(
                        "DEEPSEEK_CLASSIFICATION_MODEL", "deepseek-chat"
                    ),
                    "optimization": os.getenv(
                        "DEEPSEEK_OPTIMIZATION_MODEL", "deepseek-chat"
                    ),
                },
            )

        # Qwen 配置
        qwen_api_key = os.getenv("QWEN_API_KEY", "")
        qwen_base_url = os.getenv(
            "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/api/v1"
        )
        if qwen_api_key:
            providers["qwen"] = LLMProviderConfig(
                provider_id="qwen",
                api_key=qwen_api_key,
                base_url=qwen_base_url,
                models={
                    "generation": os.getenv("QWEN_GENERATION_MODEL", "qwen-max"),
                    "classification": os.getenv(
                        "QWEN_CLASSIFICATION_MODEL", "qwen-plus"
                    ),
                    "optimization": os.getenv("QWEN_OPTIMIZATION_MODEL", "qwen-plus"),
                },
            )

        # Ollama 配置（本地部署，api_key 可为空）
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        ollama_api_key = os.getenv(
            "OLLAMA_API_KEY", "ollama"
        )  # Ollama 通常不需要 API Key
        providers["ollama"] = LLMProviderConfig(
            provider_id="ollama",
            api_key=ollama_api_key,
            base_url=ollama_base_url,
            models={
                "generation": os.getenv("OLLAMA_GENERATION_MODEL", "qwen2.5:7b"),
                "classification": os.getenv(
                    "OLLAMA_CLASSIFICATION_MODEL", "qwen2.5:3b"
                ),
                "optimization": os.getenv("OLLAMA_OPTIMIZATION_MODEL", "qwen2.5:7b"),
            },
        )

        # 设置默认提供商
        if not self.llm.default_provider:
            self.llm.default_provider = os.getenv("LLM_DEFAULT_PROVIDER", "deepseek")

        # 合并配置, 从环境变量加载的配置优先
        self.llm.providers = {**self.llm.providers, **providers}

    def _create_directories(self):
        """创建必要的目录结构"""
        dirs_to_create = [
            self.paths.data_dir,
            self.paths.model_dir,
            self.paths.log_dir,
            self.paths.processed_data_dir,
            self.logging.file_path.parent,
        ]

        for directory in dirs_to_create:
            if isinstance(directory, Path):
                directory.mkdir(parents=True, exist_ok=True)


# 单例模式
settings = Settings()

# 调试代码：直接运行 python app/core/config.py 可以检查路径对不对
if __name__ == "__main__":
    print("=" * 60)
    print(f"项目名称：{settings.project_name} v{settings.version}")
    print("=" * 60)

    # 显示关键路径
    print("\n📁 关键路径:")
    print(f"  项目根目录：{settings.paths.project_root}")
    print(f"  数据目录：{settings.paths.data_dir}")
    print(f"  模型目录：{settings.paths.model_dir}")
    print(f"  日志目录：{settings.paths.log_dir}")

    # 显示模型配置
    print("\n🤖 模型配置:")
    print(f"  Embedding 模型：{settings.models.embedding_model}")
    print(f"  模型路径：{settings.models.embedding_model_path}")

    # 显示向量数据库配置
    print("\n🗄️ 向量数据库配置:")
    print(f"  向量维度：{settings.vectordb.vector_dimension}")

    # 显示检索器配置
    print("\n🔍 检索器配置:")
    print(f"  阈值策略：{settings.retriever.threshold_strategy}")
    print(f"  最小相似度阈值：{settings.retriever.min_similarity}")
    print(
        f"  重排权重：S={settings.retriever.weight_similarity}, "
        f"R={settings.retriever.weight_recency}, "
        f"L={settings.retriever.weight_length}"
    )

    # 显示 LLM 提供商配置
    print("\n🤖 LLM 提供商配置:")
    print(f"  默认提供商：{settings.llm.default_provider}")
    for provider_id, provider_config in settings.llm.providers.items():
        print(f"  - {provider_id}: {provider_config.models}")
