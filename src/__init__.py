"""
GovPulse - 泸州市政务智能问答系统 (Agentic RAG)

提供统一的公共 API 接口
"""

__version__ = "0.1.0"
__author__ = "GovPulse Team"

# Agent 层
from src.app.agents.rag_agent import RagAgent, query_agentic_rag
from src.app.agents.models.agent_decision import AgentDecision, AgentDecisionType, RetrievalStrategy

# LLM 服务（已重构）
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
    # Agent 层
    "RagAgent",
    "AgenticRAGEngine",  # 兼容别名
    "query_agentic_rag",
    "AgentDecision",
    "AgentDecisionType",
    "RetrievalStrategy",

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
    "retrieve_with_details",
    "get_retriever_instance",
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
