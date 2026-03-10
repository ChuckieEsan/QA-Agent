"""Agent 模块 - 基于 LangGraph 的 MultiAgent 实现"""

from .graph import create_agent_graph, agent_graph, run_agent
from .state import AgentState, ProcessStatus

# 兼容旧接口
ainvoke = run_agent

__all__ = [
    "create_agent_graph",
    "agent_graph",
    "run_agent",
    "ainvoke",
    "AgentState",
    "ProcessStatus",
]