"""Agent 状态定义 - 基于 LangGraph 的状态管理"""

from typing import TypedDict, List, Optional, Dict, Any
from enum import Enum


class ProcessStatus(str, Enum):
    """处理状态枚举"""
    PENDING = "pending"
    PREPROCESSED = "preprocessed"
    TOOLS_CALLED = "tools_called"
    FUSED = "fused"
    GENERATED = "generated"
    VALIDATED = "validated"
    COMPLETED = "completed"
    WORK_ORDER_CREATED = "work_order_created"
    FAILED = "failed"


class AgentState(TypedDict):
    """
    Agent 状态定义

    包含整个处理流程中各阶段的数据
    """
    # 原始输入
    original_query: str

    # 预处理阶段
    cleaned_query: str
    classification: Optional[Dict[str, Any]]  # 类型、紧急度
    political_elements: Optional[Dict[str, Any]]  # 五大核心要素

    # 工具调用阶段
    tool_results: List[Dict[str, Any]]  # 工具返回结果
    retrieved_knowledge: List[Dict[str, Any]]  # 检索到的知识

    # 知识融合阶段
    fused_context: str  # 融合后的上下文

    # 生成阶段
    generated_response: str

    # 验证阶段
    confidence_score: float  # 置信度

    # 元数据
    status: ProcessStatus
    error_message: Optional[str]


def create_initial_state(query: str) -> AgentState:
    """创建初始状态"""
    return {
        "original_query": query,
        "cleaned_query": "",
        "classification": None,
        "political_elements": None,
        "tool_results": [],
        "retrieved_knowledge": [],
        "fused_context": "",
        "generated_response": "",
        "confidence_score": 0.0,
        "status": ProcessStatus.PENDING,
        "error_message": None,
    }