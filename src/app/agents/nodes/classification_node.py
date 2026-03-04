"""
分类 Node - 诉求分类和部门匹配

负责：
1. 诉求类型分类（建议/投诉/求助/咨询）
2. 紧急程度分级
3. 办理部门匹配
"""

from typing import Dict, Any
from src.app.agents.state import AppealState
from src.app.infra.utils.logger import get_logger
from src.app.agents.tools import classify

logger = get_logger(__name__)


def classify_appeal(state: AppealState) -> AppealState:
    """
    分类节点 - 对诉求进行分类和部门匹配

    Args:
        state: 当前状态

    Returns:
        更新后的状态
    """
    logger.info("[ClassificationNode] 开始分类诉求")

    query = state.get("cleaned_query", "")

    if not query:
        state["error_message"] = "清洗后的诉作文本为空"
        state["current_step"] = "classification_failed"
        return state

    try:
        # 使用分类工具
        result = classify.invoke({"query": query})

        state["appeal_type"] = result.get("type", "未知")
        state["urgency_level"] = result.get("urgency_level", "一般")
        state["department"] = result.get("department", "未知")
        state["is_invalid"] = result.get("is_invalid", False)

        logger.info(
            f"[ClassificationNode] 分类完成："
            f"类型={state['appeal_type']}, "
            f"紧急程度={state['urgency_level']}, "
            f"部门={state['department']}"
        )

        state["current_step"] = "classification_completed"

    except Exception as e:
        logger.error(f"[ClassificationNode] 分类失败：{e}")
        state["error_message"] = str(e)
        state["current_step"] = "classification_failed"

    return state


def check_invalid_appeal(state: AppealState) -> Dict[str, str]:
    """
    检查是否为无效诉求

    用于条件边，决定下一步走向
    """
    if state.get("is_invalid", False):
        return {"next": "end"}
    return {"next": "continue"}
