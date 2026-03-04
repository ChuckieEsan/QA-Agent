"""
生成 Node - 基于检索结果生成回答

负责：
1. 基于检索结果生成专业回答
2. 确保回答符合政务规范
"""

from typing import Dict, Any
from src.app.agents.state import AppealState
from src.app.infra.utils.logger import get_logger
from src.app.agents.tools import generate_answer

logger = get_logger(__name__)


def generate_response(state: AppealState) -> AppealState:
    """
    生成节点 - 基于检索结果生成政务问答回复

    Args:
        state: 当前状态

    Returns:
        更新后的状态
    """
    logger.info("[GenerationNode] 开始生成回答")

    query = state.get("cleaned_query", "")
    retrieval_results = state.get("retrieval_results", [])

    if not query:
        state["error_message"] = "查询为空"
        state["current_step"] = "generation_failed"
        return state

    try:
        # 构建上下文
        context = build_context(retrieval_results)

        # 使用生成工具
        result = generate_answer.invoke({
            "query": query,
            "context": context
        })

        state["generated_answer"] = result.get("answer", "")

        logger.info(f"[GenerationNode] 生成完成：{state['generated_answer'][:50]}...")

        state["current_step"] = "generation_completed"

    except Exception as e:
        logger.error(f"[GenerationNode] 生成失败：{e}")
        state["error_message"] = str(e)
        state["current_step"] = "generation_failed"
        state["generated_answer"] = "抱歉，生成回答时出现错误，请稍后再试。"

    return state


def build_context(retrieval_results: list) -> str:
    """
    构建检索上下文文本

    Args:
        retrieval_results: 检索结果列表

    Returns:
        拼接的上下文文本
    """
    if not retrieval_results:
        return ""

    context_parts = []

    for i, result in enumerate(retrieval_results[:5], 1):
        title = result.get("title", "无标题")
        department = result.get("department", "未知部门")
        content = result.get("content", "")

        if content:
            context_parts.append(f"[参考案例 {i}]\n部门：{department}\n标题：{title}\n内容：{content}")

    return "\n\n".join(context_parts)
