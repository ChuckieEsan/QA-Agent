"""
GovPulse - 泸州市政务智能问答系统 (Agentic RAG)

提供统一的公共 API 接口
"""

__version__ = "0.1.0"
__author__ = "GovPulse Team"

# 核心引擎
from src.app.services.rag_engine import (
    AgenticRAGEngine,
    query_agentic_rag,
    query_rag,
)

# LLM 服务
from src.app.services.llm_service import (
    LLMService,
    AgentDecision,
    get_llm_service,
    generate_agentic_rag_response,
)

# 检索器
from src.app.components.retrievers.hybrid_retriever import (
    HybridVectorRetriever,
    retrieve_with_details,
    get_retriever_instance,
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
    # 核心引擎
    "AgenticRAGEngine",
    "query_agentic_rag",
    "query_rag",

    # LLM 服务
    "LLMService",
    "AgentDecision",
    "get_llm_service",
    "generate_agentic_rag_response",

    # 检索器
    "HybridVectorRetriever",
    "retrieve_with_details",
    "get_retriever_instance",

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
