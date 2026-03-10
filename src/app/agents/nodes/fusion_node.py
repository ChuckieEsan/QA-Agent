"""知识融合节点 - 多源知识去重、排序、摘要"""

from typing import Dict, Any, List
from src.app.agents.state import AgentState, ProcessStatus
from src.app.infra.llm import create_llm_service
from src.app.infra.utils.logger import get_logger

logger = get_logger(__name__)


def fusion_node(state: AgentState) -> AgentState:
    """
    知识融合节点

    将工具调用结果和知识检索结果进行融合：
    1. 去重
    2. 排序
    3. 摘要
    """
    logger.info(f"[Fusion] 开始知识融合...")

    # 收集所有知识来源
    knowledge_sources = []

    # 1. 添加工具调用结果
    for tool_result in state.get("tool_results", []):
        tool_name = tool_result.get("tool", "")
        result_data = tool_result.get("result", {})

        if tool_name == "query_knowledge_base":
            # 从知识库检索结果中提取
            if isinstance(result_data, dict):
                content = result_data.get("content", "")
                if content:
                    knowledge_sources.append({
                        "source": "knowledge_base",
                        "content": content,
                        "score": result_data.get("score", 0.5),
                    })

    # 2. 添加检索器结果
    for item in state.get("retrieved_knowledge", []):
        content = item.get("content", "")
        if content:
            knowledge_sources.append({
                "source": "milvus",
                "title": item.get("title", ""),
                "content": content,
                "score": item.get("composite_score", item.get("similarity", 0)),
                "department": item.get("department", ""),
            })

    # 3. 去重（基于内容相似度）
    unique_knowledge = deduplicate_knowledge(knowledge_sources)

    # 4. 排序（按评分）
    unique_knowledge.sort(key=lambda x: x.get("score", 0), reverse=True)

    # 5. 构建融合上下文
    if unique_knowledge:
        fused_context = build_fused_context(unique_knowledge)
    else:
        fused_context = "未找到相关知识。"

    state["fused_context"] = fused_context
    state["status"] = ProcessStatus.FUSED

    logger.info(f"[Fusion] 知识融合完成，共 {len(unique_knowledge)} 条唯一知识")

    return state


def deduplicate_knowledge(knowledge: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    知识去重

    简单基于内容完全相同进行去重
    """
    seen_content = set()
    unique = []

    for item in knowledge:
        content = item.get("content", "")
        if content and content not in seen_content:
            seen_content.add(content)
            unique.append(item)

    return unique


def build_fused_context(knowledge: List[Dict[str, Any]]) -> str:
    """
    构建融合后的上下文

    格式：
    【来源1】标题/部门
    内容摘要...

    【来源2】标题/部门
    内容摘要...
    """
    context_parts = []

    for i, item in enumerate(knowledge, 1):
        source = item.get("source", "unknown")
        title = item.get("title", "")
        department = item.get("department", "")
        content = item.get("content", "")

        # 构建来源标识
        if source == "milvus" and title:
            source_label = title
        elif source == "milvus" and department:
            source_label = department
        elif source == "knowledge_base":
            source_label = "知识库"
        else:
            source_label = source

        # 截取内容（如果太长）
        if len(content) > 500:
            content = content[:500] + "..."

        context_parts.append(f"【来源{i}】{source_label}\n{content}")

    return "\n\n".join(context_parts)