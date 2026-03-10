"""生成节点 - 回复生成和置信度评估"""

from typing import Dict, Any, List
from src.app.agents.state import AgentState, ProcessStatus
from src.app.infra.llm import create_llm_service
from src.app.components.quality import GovAnswerValidator
from src.app.infra.utils.logger import get_logger

logger = get_logger(__name__)


# 置信度阈值
CONFIDENCE_THRESHOLD = 0.6


def generate_node(state: AgentState) -> AgentState:
    """
    生成节点

    使用 LLM 生成回复
    """
    query = state["original_query"]
    context = state.get("fused_context", "")
    classification = state.get("classification", {})
    political_elements = state.get("political_elements", {})

    logger.info(f"[Generate] 开始生成回复...")

    llm_service = create_llm_service()

    # 构建系统提示词
    system_prompt = f"""你是政务问政智能回复助手。请根据以下信息生成回复。

## 市民诉求
{query}

## 诉求分类
- 类型: {classification.get('request_type', '未知')}
- 紧急程度: {classification.get('request_urgency', '未知')}

## 五大核心要素
- 时间: {political_elements.get('time', '未知')}
- 地点: {political_elements.get('location', '未知')}
- 事件: {political_elements.get('event', '未知')}
- 诉求目标: {political_elements.get('goal', '未知')}
- 涉及主体: {', '.join(political_elements.get('subjects', []))}

## 相关知识
{context}

## 回复要求
1. 回复内容要准确、完整、简洁
2. 如果知识库中有相关信息，请基于信息回答
3. 如果没有相关信息，请告知市民可能的办理渠道
4. 体现政务服务的专业性和亲和力
5. 字数控制在 200-500 字之间
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "请生成回复"},
    ]

    try:
        response = llm_service.generate(messages, temperature=0.7, max_tokens=1000)
        state["generated_response"] = response
        logger.info(f"[Generate] 回复生成完成，长度: {len(response)} 字符")

    except Exception as e:
        logger.error(f"[Generate] 回复生成失败: {e}")
        state["generated_response"] = "抱歉，系统暂时无法生成回复，请稍后重试或拨打12345热线。"
        state["status"] = ProcessStatus.FAILED
        state["error_message"] = str(e)
        return state

    state["status"] = ProcessStatus.GENERATED
    return state


async def validate_node(state: AgentState) -> AgentState:
    """
    验证节点

    使用 GovAnswerValidator 评估回复质量
    """
    query = state["original_query"]
    answer = state.get("generated_response", "")
    context = state.get("fused_context", "")

    logger.info(f"[Validate] 开始置信度评估...")

    try:
        validator = GovAnswerValidator()
        validation_result = await validator.validate(answer, query, context)

        confidence_score = validation_result.get("overall_score", 0.0)
        state["confidence_score"] = confidence_score

        logger.info(f"[Validate] 置信度: {confidence_score:.2f}")

    except Exception as e:
        logger.warning(f"[Validate] 置信度评估失败: {e}")
        # 默认给一个中等置信度
        state["confidence_score"] = 0.5

    state["status"] = ProcessStatus.VALIDATED

    return state


def should_auto_reply(state: AgentState) -> str:
    """
    判断是否自动回复

    根据置信度决定后续流程：
    - 置信度 >= 阈值: 自动回复 (COMPLETED)
    - 置信度 < 阈值: 创建工单 (WORK_ORDER_CREATED)
    """
    confidence = state.get("confidence_score", 0.0)

    if confidence >= CONFIDENCE_THRESHOLD:
        logger.info(f"[Decision] 置信度 {confidence:.2f} >= {CONFIDENCE_THRESHOLD}，自动回复")
        state["status"] = ProcessStatus.COMPLETED
        return "auto_reply"
    else:
        logger.info(f"[Decision] 置信度 {confidence:.2f} < {CONFIDENCE_THRESHOLD}，创建工单")
        state["status"] = ProcessStatus.WORK_ORDER_CREATED
        return "create_work_order"