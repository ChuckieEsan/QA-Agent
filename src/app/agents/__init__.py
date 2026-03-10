"""
Agent 模块 - 基于 LangGraph 的 MultiAgent 实现

提供政务问政智能回复的完整工作流：
- 诉求预处理（清洗、脱敏、分类）
- 五大核心要素提取
- 工具调用（MCP 工具）
- 知识检索（混合向量检索 + BGE 重排）
- 知识融合
- 回复生成
- 置信度评估

示例用法：
    from src.app.agents import run_agent, ainvoke

    # 异步调用
    result = await ainvoke("请问灵活就业人员如何缴纳社保？")

    # 或
    result = await run_agent("请问灵活就业人员如何缴纳社保？")
"""

from .graph import create_agent_graph, agent_graph, run_agent
from .state import AgentState, ProcessStatus, create_initial_state
from .tools import get_available_tools, ToolRegistry

# 兼容旧接口
ainvoke = run_agent

# 同步调用兼容（可选）
def invoke(query: str) -> AgentState:
    """
    同步调用 Agent

    Args:
        query: 用户查询

    Returns:
        AgentState: 执行结果状态
    """
    import asyncio
    return asyncio.run(run_agent(query))

__all__ = [
    # 核心接口
    "create_agent_graph",
    "agent_graph",
    "run_agent",
    "ainvoke",
    "invoke",
    # 状态
    "AgentState",
    "ProcessStatus",
    "create_initial_state",
    # 工具
    "get_available_tools",
    "ToolRegistry",
]