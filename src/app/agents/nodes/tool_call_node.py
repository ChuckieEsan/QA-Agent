"""工具调用节点 - MCP 工具调用和知识检索"""

from typing import Dict, Any, List
from src.app.agents.state import AgentState, ProcessStatus
from src.app.agents.tools.registry import ToolRegistry
from src.app.components.retrievers import HybridVectorRetriever
from src.app.infra.utils.logger import get_logger

logger = get_logger(__name__)


def tool_call_node(state: AgentState) -> AgentState:
    """
    工具调用节点

    调用 MCP 工具（工单创建、知识库查询等）
    """
    query = state["cleaned_query"]
    classification = state.get("classification", {})
    political_elements = state.get("political_elements", {})

    logger.info(f"[ToolCall] 开始工具调用...")

    # 获取工具注册表
    registry = ToolRegistry()

    tool_results = []

    # 1. 根据诉求类型决定调用哪些工具
    request_type = classification.get("request_type", "consult")

    # 咨询类诉求 - 检索知识库
    if request_type == "consult":
        try:
            # 调用知识库检索工具
            result = registry.call_tool(
                "query_knowledge_base",
                {"query": query, "top_k": 5}
            )
            tool_results.append({
                "tool": "query_knowledge_base",
                "result": result,
            })
            logger.info(f"[ToolCall] 知识库检索完成")
        except Exception as e:
            logger.warning(f"[ToolCall] 知识库检索失败: {e}")

    # 2. 查询权责清单（可选）
    if political_elements.get("subjects"):
        try:
            for subject in political_elements["subjects"][:3]:  # 最多3个
                result = registry.call_tool(
                    "query_department_responsibility",
                    {"department": subject}
                )
                tool_results.append({
                    "tool": "query_department_responsibility",
                    "subject": subject,
                    "result": result,
                })
        except Exception as e:
            logger.warning(f"[ToolCall] 权责清单查询失败: {e}")

    state["tool_results"] = tool_results
    state["status"] = ProcessStatus.TOOLS_CALLED

    return state


def knowledge_retrieval_node(state: AgentState) -> AgentState:
    """
    知识检索节点

    复用现有的 HybridVectorRetriever 进行知识检索
    """
    query = state["cleaned_query"]
    logger.info(f"[KnowledgeRetrieval] 开始知识检索: {query[:50]}...")

    # 复用现有的检索器
    retriever = HybridVectorRetriever()

    try:
        context, results, metadata = retriever.retrieve(query, top_k=5)

        state["retrieved_knowledge"] = results

        logger.info(f"[KnowledgeRetrieval] 检索到 {len(results)} 条知识")

    except Exception as e:
        logger.warning(f"[KnowledgeRetrieval] 检索失败: {e}")
        state["retrieved_knowledge"] = []

    state["status"] = ProcessStatus.TOOLS_CALLED

    return state