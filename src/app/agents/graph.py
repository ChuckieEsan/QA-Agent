"""
LangGraph 工作流定义 - 政务问答智能代理

包含 4 个主节点和 2 个条件路由：
- classify: 分类与意图识别
- retrieve: 双路召回（权责清单 + 历史案例）
- generate: 知识融合与回复生成
- action: 兜底工单创建
"""

import asyncio
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from typing import Optional, Any

from src.app.agents.state import AgentState
from src.app.components.classifier.request_classifier import GovRequestClassifier, GovRequestType, create_gov_request_classifier
from src.app.components.retriever.powers_retriever import create_powers_retriever
from src.app.components.retriever.cases_retriever import create_cases_retriever
from src.app.infra.llm import BaseLLMService, create_llm_service
from src.app.infra.utils.logger import get_logger
from src.app.agents.tools.mcp_tools import create_work_order

logger = get_logger(__name__)


# ==================== 组件初始化 ====================

def _init_classifier() -> GovRequestClassifier:
    """初始化分类器"""
    llm = create_llm_service(provider_id="deepseek", model_name="deepseek-chat")
    return create_gov_request_classifier(llm=llm)


def _init_retrievers():
    """初始化检索器"""
    powers_retriever = create_powers_retriever(top_k=3)
    cases_retriever = create_cases_retriever(top_k=3)
    return powers_retriever, cases_retriever


def _init_llm() -> BaseLLMService:
    """初始化 LLM 服务"""
    return create_llm_service(provider_id="deepseek", model_name="deepseek-chat")


# 惰性初始化组件
_classifier: Optional[GovRequestClassifier] = None
_powers_retriever = None
_cases_retriever = None
_llm = None


def get_classifier() -> GovRequestClassifier:
    """获取分类器（惰性初始化）"""
    global _classifier
    if _classifier is None:
        _classifier = _init_classifier()
    return _classifier


def get_powers_retriever():
    """获取权责清单检索器"""
    global _powers_retriever
    if _powers_retriever is None:
        _powers_retriever, _ = _init_retrievers()
    return _powers_retriever


def get_cases_retriever():
    """获取案例检索器"""
    global _cases_retriever
    if _cases_retriever is None:
        _, _cases_retriever = _init_retrievers()
    return _cases_retriever


def get_llm():
    """获取 LLM 服务"""
    global _llm
    if _llm is None:
        _llm = _init_llm()
    return _llm


# ==================== 节点实现 ====================

def classify_node(state: AgentState) -> dict:
    """
    节点 1: 分类与意图识别

    调用 Classifier 解析用户意图和部门
    """
    latest_msg = state["messages"][-1].content
    logger.info(f"[Classify] 处理查询: {latest_msg[:50]}...")

    try:
        classifier = get_classifier()
        result = classifier.classify(latest_msg)

        logger.info(f"[Classify] 分类结果: {result.request_type}")

        return {
            "classification": result.model_dump() if hasattr(result, 'model_dump') else result
        }
    except Exception as e:
        logger.error(f"[Classify] 分类失败: {e}")
        # 失败时返回默认分类
        return {
            "classification": {
                "request_type": "consult",
                "request_department": ""
            }
        }


def retrieve_node(state: AgentState) -> dict:
    """
    节点 2: 双路召回

    从权责清单和历史案例中检索相关信息
    """
    latest_msg = state["messages"][-1].content
    classification = state.get("classification", {}) or {}

    logger.info(f"[Retrieve] 开始双路召回: {latest_msg[:50]}...")

    try:
        # 检索权责清单
        powers_retriever = get_powers_retriever()
        powers_docs = powers_retriever._get_relevant_documents(latest_msg)

        # 检索历史案例
        cases_retriever = get_cases_retriever()
        cases_docs = cases_retriever._get_relevant_documents(latest_msg)

        # 构建上下文
        context_parts = []

        if powers_docs:
            context_parts.append("【权责清单】")
            for i, doc in enumerate(powers_docs[:2], 1):
                context_parts.append(f"{i}. {doc.page_content[:200]}")

        if cases_docs:
            context_parts.append("\n【历史案例】")
            for i, doc in enumerate(cases_docs[:2], 1):
                context_parts.append(f"{i}. 问: {doc.metadata.get('question', '')[:100]}")
                context_parts.append(f"   答: {doc.metadata.get('answer', '')[:150]}")

        context = "\n".join(context_parts) if context_parts else "未找到相关内容"

        logger.info(f"[Retrieve] 召回完成，上下文长度: {len(context)}")

        return {"retrieved_context": context}
    except Exception as e:
        logger.error(f"[Retrieve] 检索失败: {e}")
        return {"retrieved_context": "检索服务暂时不可用，请稍后再试。"}


def generate_node(state: AgentState) -> dict:
    """
    节点 3: 知识融合与回复生成

    使用 LLM 生成回复，并输出置信度评分
    """
    from langchain_core.prompts import ChatPromptTemplate
    from pydantic import BaseModel

    latest_msg = state["messages"][-1].content
    context = state.get("retrieved_context", "")

    logger.info(f"[Generate] 开始生成回复，上下文长度: {len(context)}")

    # 定义置信度输出模型
    class GenerationResult(BaseModel):
        reply: str
        confidence: float

    try:
        llm = get_llm()

        # 构建提示词
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一位政务问答助手，请根据提供的上下文信息回答用户的问题。

要求：
1. 只根据上下文内容回答，不要编造信息
2. 如果上下文没有相关信息，请如实说明"抱歉，我在知识库中未找到相关内容"
3. 回答要简洁明了，使用规范的口语化表达

上下文信息：
{context}
"""),
            ("user", "{query}")
        ])

        # 构建 LCEL 链
        chain = prompt | llm.with_structured_output(GenerationResult, method="function_calling")

        # 调用生成
        result = chain.invoke(
            {"query": latest_msg, "context": context},
            config={"temperature": 0.3, "max_tokens": 500}
        )

        logger.info(f"[Generate] 生成完成，置信度: {result.confidence}")

        return {
            "final_reply": result.reply,
            "confidence_score": result.confidence
        }
    except Exception as e:
        logger.error(f"[Generate] 生成失败: {e}")

        # 降级处理：直接返回上下文中的内容
        fallback_reply = context[:300] if context else "抱歉，服务暂时不可用。"
        return {
            "final_reply": fallback_reply,
            "confidence_score": 0.3
        }


def action_node(state: AgentState) -> dict:
    """
    节点 4: 兜底动作 - 调用 MCP 创建工单

    当置信度低于阈值或用户情绪逃逸时触发
    """
    latest_msg = state["messages"][-1].content
    classification = state.get("classification", {}) or {}
    request_type = classification.get("request_type", "咨询")

    logger.info(f"[Action] 触发兜底工单，请求类型: {request_type}")

    try:
        # 调用 MCP 工具创建工单
        order_data = {
            "title": latest_msg[:50],
            "content": latest_msg,
            "request_type": request_type,
        }

        result = create_work_order.invoke(order_data)
        work_order_id = result.get("order_id", "UNKNOWN")

        fallback_reply = (
            f"抱歉，由于您反映的问题较为复杂，系统已自动为您生成政务工单（单号：{work_order_id}），"
            f"将由相关部门专人为您跟进处理。"
        )

        logger.info(f"[Action] 工单创建成功: {work_order_id}")

        return {
            "final_reply": fallback_reply,
            "work_order_id": work_order_id
        }
    except Exception as e:
        logger.error(f"[Action] 工单创建失败: {e}")
        return {
            "final_reply": "抱歉，您的问题已记录，我们将尽快与您联系。",
            "work_order_id": None
        }


# ==================== 条件路由 ====================

def escalation_router(state: AgentState) -> str:
    """
    情绪逃逸/风控路由

    规则：
    - 对话超过 6 条消息（3轮），说明没解决问题，直接兜底
    - 投诉举报类型直接派单
    """
    messages = state.get("messages", [])
    classification = state.get("classification", {}) or {}

    # 规则1: 对话轮数过多
    if len(messages) >= 6:
        logger.info("[Escalation] 对话轮数过多，触发兜底")
        return "action"

    # 规则2: 投诉举报类型直接派单
    request_type = classification.get("request_type", "")
    if request_type in ["complaint", "投诉"]:
        logger.info("[Escalation] 投诉类型，触发兜底")
        return "action"

    return "retrieve"


def confidence_router(state: AgentState) -> str:
    """
    置信度路由

    规则：
    - 置信度 < 0.45 → 触发兜底工单
    - 否则 → 正常结束
    """
    confidence = state.get("confidence_score", 1.0)

    if confidence < 0.45:
        logger.info(f"[Confidence] 置信度 {confidence} < 0.45，触发兜底")
        return "action"

    logger.info(f"[Confidence] 置信度 {confidence} >= 0.45，正常返回")
    return END


# ==================== 构建 Graph ====================

def create_graph():
    """构建 LangGraph 工作流"""
    builder = StateGraph(AgentState)

    # 添加节点
    builder.add_node("classify", classify_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.add_node("action", action_node)

    # 设置边
    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "classify",
        escalation_router,
        {"action": "action", "retrieve": "retrieve"}
    )
    builder.add_edge("retrieve", "generate")
    builder.add_conditional_edges(
        "generate",
        confidence_router,
        {"action": "action", END: END}
    )
    builder.add_edge("action", END)

    return builder


# ==================== 编译 Graph ====================

# 使用 MemorySaver 实现会话状态持久化
memory = MemorySaver()
gov_agent_app = create_graph().compile(checkpointer=memory)

logger.info("[Graph] LangGraph 工作流编译完成")


# ==================== 对外接口 ====================

async def ainvoke(query: str, session_id: str = "default") -> dict:
    """
    异步调用接口

    Args:
        query: 用户查询
        session_id: 会话ID（用于多轮对话状态）

    Returns:
        最终状态字典
    """

    config = {"configurable": {"thread_id": session_id}}
    inputs = {"messages": [HumanMessage(content=query)]}

    result_state = gov_agent_app.invoke(inputs, config=config)

    return result_state


def invoke(query: str, session_id: str = "default") -> dict:
    """
    同步调用接口

    Args:
        query: 用户查询
        session_id: 会话ID（用于多轮对话状态）

    Returns:
        最终状态字典
    """
    return asyncio.run(ainvoke(query, session_id))