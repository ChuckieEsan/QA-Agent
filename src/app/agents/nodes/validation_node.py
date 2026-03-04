"""
验证 Node - 回答质量验证

负责：
1. 验证回答是否符合政务规范
2. 验证不通过时触发重新生成
"""

from typing import Dict, Any
from src.app.agents.state import AppealState
from src.app.infra.utils.logger import get_logger
from src.app.agents.tools import validate_answer

logger = get_logger(__name__)


def validate_response(state: AppealState) -> AppealState:
    """
    验证节点 - 验证生成的回答是否符合政务规范

    Args:
        state: 当前状态

    Returns:
        更新后的状态
    """
    logger.info("[ValidationNode] 开始验证回答")

    answer = state.get("generated_answer", "")
    query = state.get("cleaned_query", "")
    retrieval_results = state.get("retrieval_results", [])

    if not answer:
        state["error_message"] = "生成的回答为空"
        state["current_step"] = "validation_failed"
        return state

    try:
        # 构建上下文
        context_parts = []
        for result in retrieval_results[:3]:
            content = result.get("content", "")
            if content:
                context_parts.append(content)
        context = "\n\n".join(context_parts)

        # 使用验证工具
        result = validate_answer.invoke({
            "answer": answer,
            "query": query,
            "context": context
        })

        state["validation_result"] = result

        overall_score = result.get("overall_score", 0)
        passed = result.get("passed", False)

        logger.info(
            f"[ValidationNode] 验证完成："
            f"综合评分={overall_score:.2f}, 通过={passed}"
        )

        if passed:
            state["final_response"] = answer
            state["current_step"] = "validation_passed"
        else:
            # 验证未通过，保留验证结果用于重新生成
            state["current_step"] = "validation_failed_retry"

    except Exception as e:
        logger.error(f"[ValidationNode] 验证失败：{e}")
        state["error_message"] = str(e)
        state["current_step"] = "validation_failed"
        # 验证失败时，仍然返回生成的回答
        state["final_response"] = answer

    return state


def check_validation_result(state: AppealState) -> Dict[str, str]:
    """
    检查验证结果

    用于条件边，决定是结束还是重新生成
    """
    validation_result = state.get("validation_result", {})
    passed = validation_result.get("passed", False)

    if passed:
        return {"next": "end"}
    else:
        return {"next": "regenerate"}
