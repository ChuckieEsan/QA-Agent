"""
政务问答智能代理模块

基于 LangGraph StateGraph 的工作流实现
"""

from .react_agent import (
    gov_agent_app,
)

__all__ = [
    "gov_agent_app",
]