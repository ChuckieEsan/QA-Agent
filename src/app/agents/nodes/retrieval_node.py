"""
检索 Node - 知识库向量检索

负责：
1. 基于诉求内容检索相关案例
2. 返回检索结果和上下文
"""

from typing import Dict, Any, List
from src.app.agents.state import AppealState
from src.app.infra.utils.logger import get_logger
from src.app.agents.tools import retrieve

logger = get_logger(__name__)


def retrieve_context(state: AppealState) -> AppealState:
    """
    检索节点 - 检索知识库中的相关案例

    Args:
        state: 当前状态

    Returns:
        更新后的状态
    """
    logger.info("[RetrievalNode] 开始检索知识库")

    query = state.get("cleaned_query", "")

    if not query:
        state["error_message"] = "检索 query 为空"
        state["current_step"] = "retrieval_failed"
        return state

    try:
        # 使用检索工具
        result = retrieve.invoke({
            "query": query,
            "top_k": 5,
            "threshold": 0.5
        })

        state["retrieval_results"] = result.get("results", [])

        logger.info(
            f"[RetrievalNode] 检索完成："
            f"找到 {len(state['retrieval_results'])} 个结果"
        )

        state["current_step"] = "retrieval_completed"

    except Exception as e:
        logger.error(f"[RetrievalNode] 检索失败：{e}")
        state["error_message"] = str(e)
        state["current_step"] = "retrieval_failed"
        # 即使检索失败，也继续流程，让生成节点处理
        state["retrieval_results"] = []

    return state


def check_retrieval_results(state: AppealState) -> Dict[str, str]:
    """
    检查检索结果质量

    用于条件边，决定是否需要重新检索或直接生成
    """
    results = state.get("retrieval_results", [])

    if len(results) == 0:
        return {"next": "generate_with_no_context"}

    # 检查最高相似度
    max_similarity = max(r.get("similarity", 0) for r in results)

    if max_similarity < 0.4:
        return {"next": "generate_with_no_context"}

    return {"next": "generate"}
