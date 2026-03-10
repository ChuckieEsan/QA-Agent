"""LangGraph 图定义 - MultiAgent 工作流"""

from langgraph.graph import StateGraph, END
from src.app.agents.state import AgentState, ProcessStatus, create_initial_state
from src.app.agents.nodes import (
    preprocess_node,
    extract_elements_node,
    tool_call_node,
    knowledge_retrieval_node,
    fusion_node,
    generate_node,
    validate_node,
)
from src.app.infra.utils.logger import get_logger

logger = get_logger(__name__)


def should_auto_reply(state: AgentState) -> str:
    """
    判断是否自动回复

    根据置信度决定后续流程：
    - 置信度 >= 0.6: 自动回复 (COMPLETED)
    - 置信度 < 0.6: 创建工单 (WORK_ORDER_CREATED)
    """
    confidence = state.get("confidence_score", 0.0)

    if confidence >= 0.6:
        logger.info(f"[Decision] 置信度 {confidence:.2f} >= 0.6，自动回复")
        state["status"] = ProcessStatus.COMPLETED
        return "auto_reply"
    else:
        logger.info(f"[Decision] 置信度 {confidence:.2f} < 0.6，创建工单")
        state["status"] = ProcessStatus.WORK_ORDER_CREATED
        return "create_work_order"


def create_agent_graph():
    """
    创建 Agent 图

    处理流程：
    1. 预处理 -> 2. 要素提取 -> 3. 工具调用 -> 4. 知识检索
    -> 5. 知识融合 -> 6. 回复生成 -> 7. 置信度评估 -> 8. 决策
    """
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("preprocess", preprocess_node)
    workflow.add_node("extract_elements", extract_elements_node)
    workflow.add_node("tool_call", tool_call_node)
    workflow.add_node("knowledge_retrieval", knowledge_retrieval_node)
    workflow.add_node("fusion", fusion_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("validate", validate_node)

    # 设置入口点
    workflow.set_entry_point("preprocess")

    # 添加边（线性流程）
    workflow.add_edge("preprocess", "extract_elements")
    workflow.add_edge("extract_elements", "tool_call")
    workflow.add_edge("tool_call", "knowledge_retrieval")
    workflow.add_edge("knowledge_retrieval", "fusion")
    workflow.add_edge("fusion", "generate")
    workflow.add_edge("generate", "validate")

    # 添加条件边：置信度判断
    workflow.add_conditional_edges(
        "validate",
        should_auto_reply,
        {
            "auto_reply": END,
            "create_work_order": END,  # 创建工单后结束
        }
    )

    return workflow.compile()


# 全局单例
agent_graph = create_agent_graph()


async def run_agent(query: str) -> AgentState:
    """
    运行 Agent

    Args:
        query: 用户查询

    Returns:
        最终状态
    """
    logger.info(f"[Agent] 开始处理查询: {query[:50]}...")

    # 创建初始状态
    initial_state = create_initial_state(query)

    # 执行图
    result = await agent_graph.ainvoke(initial_state)

    logger.info(f"[Agent] 处理完成，状态: {result['status']}")

    return result