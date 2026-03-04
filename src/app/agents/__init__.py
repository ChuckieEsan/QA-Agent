"""
Agents 模块 - 政务问答 Agent 核心

基于 LangGraph 实现的多智能体协同架构

使用示例:
    from src.app.agents import invoke

    result = invoke("2024 年泸州雨露计划补贴标准是多少？")
    print(result.get("final_response"))
"""

from src.app.agents.state import AppealState
from src.app.agents.graphs.mvp_graph import (
    build_mvp_graph,
    get_mvp_graph,
    invoke,
    ainvoke,
)

__all__ = [
    "AppealState",
    "build_mvp_graph",
    "get_mvp_graph",
    "invoke",
    "ainvoke",
]
