"""
政务问答智能代理模块

基于 LangGraph StateGraph 的工作流实现
"""

from typing import Any
from langgraph.checkpoint.memory import MemorySaver

from .react_agent import (
    gov_agent_app,
)
from .schemas import AgentResponse

# 创建 checkpointer 用于维护会话状态
_memory = MemorySaver()


async def ainvoke(query: str, session_id: str = "default", **kwargs) -> AgentResponse:
    """
    异步调用政务问答智能体

    Args:
        query: 用户查询
        session_id: 会话ID，用于多轮对话状态管理
        **kwargs: 其他参数

    Returns:
        AgentResponse: 包含答案和元数据的响应模型
    """
    # 构建输入状态
    input_state = {
        "messages": [{"role": "user", "content": query}],
    }

    # 配置 config 包括 thread_id (即 session_id)
    config = {
        "configurable": {
            "thread_id": session_id,
        }
    }

    # 调用 agent
    result = await gov_agent_app.ainvoke(input_state, config)

    # 提取最终回复
    messages = result.get("messages", [])
    final_reply = ""
    if messages:
        # 获取最后一条消息的内容
        last_msg = messages[-1]
        if hasattr(last_msg, "content"):
            final_reply = last_msg.content
        else:
            final_reply = str(last_msg)

    # 提取分类信息
    classification = {}
    work_order_id = None
    confidence_score = 0.0

    # 从工具调用结果中提取信息
    for msg in messages:
        # 检查是否有工具消息
        if hasattr(msg, "type") and msg.type == "tool":
            content = msg.content
            # 尝试解析工具返回的 JSON
            if "request_type" in content:
                try:
                    import json
                    # 尝试提取 JSON
                    if "{" in content:
                        json_str = content[content.find("{"):content.rfind("}")+1]
                        tool_result = json.loads(json_str)
                        classification = {
                            "request_type": tool_result.get("request_type", "未知"),
                            "request_department": tool_result.get("request_department", ""),
                        }
                        confidence_score = tool_result.get("confidence", 0.0)
                except (json.JSONDecodeError, ValueError):
                    pass

    # 检查是否有工单创建
    for msg in messages:
        if hasattr(msg, "name") and msg.name == "create_work_order":
            work_order_id = msg.content if hasattr(msg, "content") else str(msg)

    return AgentResponse(
        messages=messages,
        final_reply=final_reply,
        classification=classification,
        work_order_id=work_order_id,
        confidence_score=confidence_score,
    )


__all__ = [
    "gov_agent_app",
    "ainvoke",
    "AgentResponse",
]