"""
GovPulse - 泸州市政务智能问答系统

基于 LangGraph 实现的多智能体协同架构
"""

__version__ = "0.3.0"
__author__ = "GovPulse Team"


# 组件层
from src.app.components import (
    BaseClassifier,
    GovRequestClassifier,
    BaseMemory,
    ConversationMemory,
    BaseRetriever,
    HybridVectorRetriever,
    BaseValidator,
    GovAnswerValidator,
)

# 设置别名以保持兼容性
AnswerValidator = GovAnswerValidator

# Agent 层
from src.app.agents import (
    run_agent,
    ainvoke,
    invoke,
    agent_graph,
    AgentState,
    ProcessStatus,
    create_initial_state,
    get_available_tools,
    ToolRegistry,
)

# 数据库客户端
from src.app.infra.db.milvus_db import (
    MilvusDBClient,
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
    # 组件
    "BaseClassifier",
    "GovRequestClassifier",
    "BaseMemory",
    "ConversationMemory",
    "BaseRetriever",
    "HybridVectorRetriever",
    "BaseValidator",
    "GovAnswerValidator",
    "AnswerValidator",

    # Agent
    "run_agent",
    "ainvoke",
    "invoke",
    "agent_graph",
    "AgentState",
    "ProcessStatus",
    "create_initial_state",
    "get_available_tools",
    "ToolRegistry",

    # 数据库
    "MilvusDBClient",
    
    
    # 配置
    "settings",
    "Settings",
    "PathConfig",
    "ModelConfig",
    "MilvusDBConfig",
    "RetrieverConfig",
    "LLMConfig",
]