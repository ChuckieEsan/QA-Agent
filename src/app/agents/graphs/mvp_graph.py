"""
MVP 版本工作流编排

基于 LangGraph StateGraph 实现的政务问答工作流

流程：
START → 预处理 → 分类 → 检索 → 生成 → 验证 → END
                                    ↑______|
                                     重试
"""

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated

from src.app.agents.state import AppealState
from src.app.agents.nodes import (
    preprocess_query,
    classify_appeal,
    retrieve_context,
    generate_response,
    validate_response,
    check_validation_result,
)
from src.app.infra.utils.logger import get_logger

logger = get_logger(__name__)


def build_mvp_graph() -> StateGraph:
    """
    构建 MVP 版本工作流

    流程说明：
    1. 预处理：文本清洗、脱敏、要素提取
    2. 分类：诉求类型分类、部门匹配
    3. 检索：向量检索相关案例
    4. 生成：基于检索结果生成回答
    5. 验证：验证回答质量，不通过则重新生成

    Returns:
        编译后的 StateGraph
    """
    # 创建状态图
    builder = StateGraph(AppealState)

    # ========== 添加节点 ==========

    # 预处理节点
    builder.add_node("preprocessing", preprocess_query)

    # 分类节点
    builder.add_node("classification", classify_appeal)

    # 检索节点
    builder.add_node("retrieval", retrieve_context)

    # 生成节点
    builder.add_node("generation", generate_response)

    # 验证节点
    builder.add_node("validation", validate_response)

    # ========== 添加边 ==========

    # START → 预处理
    builder.add_edge(START, "preprocessing")

    # 预处理 → 分类
    builder.add_edge("preprocessing", "classification")

    # 分类 → 检索
    builder.add_edge("classification", "retrieval")

    # 检索 → 生成
    builder.add_edge("retrieval", "generation")

    # 生成 → 验证
    builder.add_edge("generation", "validation")

    # ========== 条件边：验证结果 ==========

    # 验证通过后结束，验证失败则重新生成
    builder.add_conditional_edges(
        "validation",
        check_validation_route,
        {
            "end": END,
            "regenerate": "generation"
        }
    )

    # 编译
    logger.info("[MVPGraph] 编译 MVP 工作流...")
    graph = builder.compile()
    logger.info("[MVPGraph] 编译完成")

    return graph


def check_validation_route(state: AppealState) -> str:
    """
    检查验证结果，决定下一步

    Returns:
        "end" 或 "regenerate"
    """
    validation_result = state.get("validation_result", {})
    passed = validation_result.get("passed", False)

    # 验证通过或达到最大重试次数（通过 step 判断）
    if passed:
        logger.debug("[ValidationRoute] 验证通过，结束流程")
        return "end"

    # 检查重试次数（防止无限循环）
    retry_count = state.get("_retry_count", 0)
    if retry_count >= 2:
        logger.warning("[ValidationRoute] 达到最大重试次数，使用当前回答")
        # 即使验证未通过，也结束流程
        state["final_response"] = state.get("generated_answer", "")
        return "end"

    logger.info(f"[ValidationRoute] 验证未通过，重新生成 (retry={retry_count + 1})")
    state["_retry_count"] = retry_count + 1
    return "regenerate"


# 创建全局单例
_mvp_graph = None


def get_mvp_graph() -> StateGraph:
    """
    获取 MVP 工作流单例

    Returns:
        编译后的 StateGraph
    """
    global _mvp_graph
    if _mvp_graph is None:
        _mvp_graph = build_mvp_graph()
    return _mvp_graph


# 快捷调用接口
def invoke(query: str) -> AppealState:
    """
    调用 MVP 工作流处理诉求

    Args:
        query: 用户诉求文本

    Returns:
        处理后的 AppealState
    """
    graph = get_mvp_graph()
    initial_state = AppealState(raw_query=query)
    result = graph.invoke(initial_state)
    return result


async def ainvoke(query: str) -> AppealState:
    """
    异步调用 MVP 工作流处理诉求

    Args:
        query: 用户诉求文本

    Returns:
        处理后的 AppealState
    """
    graph = get_mvp_graph()
    initial_state = AppealState(raw_query=query)
    result = await graph.ainvoke(initial_state)
    return result
