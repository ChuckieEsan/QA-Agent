"""
GovPulse - 泸州市政务智能问答系统

基于 LangGraph 实现的多智能体协同架构
"""

__version__ = "0.3.0"
__author__ = "GovPulse Team"

# Agent 层 - LangGraph 多智能体协同
from src.app.agents import (
    AppealState,
    invoke,
    ainvoke,
    get_mvp_graph,
)

# LLM 服务
from src.app.infra.llm.multi_model_service import (
    get_heavy_llm_service,
    get_light_llm_service,
    get_optimizer_llm_service,
    ModelPurpose,
    get_llm_service,
)

# 组件层
from src.app.components import (
    BaseGenerator,
    LLMGenerator,
    BaseClassifier,
    GovClassifier,
    BaseMemory,
    ConversationMemory,
    BaseRetriever,
    HybridVectorRetriever,
    BaseValidator,
    AnswerValidator,
)

# 数据库客户端
from src.app.infra.db.milvus_db import (
    MilvusDBClient,
    get_milvus_client,
    get_milvus_client_from_config,
)

# 配置
from src.config.setting import settings

# 配置类（供类型提示使用）
from src.config.setting import (
    Settings,
    PathConfig,
    ModelConfig,
    MilvusDBConfig,
    RetrieverConfig,
    LLMConfig,
)

__all__ = [
    # Agent 层 - LangGraph 多智能体协同
    "AppealState",
    "invoke",
    "ainvoke",
    "get_mvp_graph",

    # LLM 服务
    "get_heavy_llm_service",
    "get_light_llm_service",
    "get_optimizer_llm_service",
    "ModelPurpose",
    "get_llm_service",

    # 组件
    "BaseGenerator",
    "LLMGenerator",
    "BaseClassifier",
    "GovClassifier",
    "BaseMemory",
    "ConversationMemory",
    "BaseRetriever",
    "HybridVectorRetriever",
    "BaseValidator",
    "AnswerValidator",

    # 数据库
    "MilvusDBClient",
    "get_milvus_client",
    "get_milvus_client_from_config",

    # 配置
    "settings",
    "Settings",
    "PathConfig",
    "ModelConfig",
    "MilvusDBConfig",
    "RetrieverConfig",
    "LLMConfig",
]
