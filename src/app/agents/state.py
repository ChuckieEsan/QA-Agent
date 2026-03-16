"""Agent 状态定义 - 基于 LangGraph 的状态管理"""

from typing import Annotated, TypedDict, Optional, Any
from langchain_core.messages import HumanMessage
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage


class AgentState(TypedDict):
    """
    Agent 状态定义 - 基于 LangGraph add_messages

    使用 LangGraph 的 add_messages 注解自动处理对话历史
    """
    # LangGraph 原生消息流，自动 append 历史记录
    messages: Annotated[list[AnyMessage], add_messages]

    # 业务流转状态
    classification: Optional[Any]  # GovRequestClassifiedResult
    retrieved_context: Optional[str]  # 检索到的上下文
    confidence_score: Optional[float]  # 置信度 0-1
    final_reply: Optional[str]  # 最终回复
    work_order_id: Optional[str]  # 工单ID（如果触发兜底）


def create_initial_state(query: str) -> AgentState:
    """创建初始状态"""

    return {
        "messages": [HumanMessage(content=query)],
        "classification": None,
        "retrieved_context": "",
        "confidence_score": 0.0,
        "final_reply": "",
        "work_order_id": None,
    }