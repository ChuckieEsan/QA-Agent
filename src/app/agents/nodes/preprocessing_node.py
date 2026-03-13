"""预处理节点 - 诉求分类和要素提取"""

import re
from typing import Dict, Any, List, TypedDict
from src.app.agents.state import AgentState, ProcessStatus
from src.app.components.classifier import GovRequestClassifier
from src.app.infra.llm import create_llm_service
from src.app.infra.llm.base_llm_service import BaseLLMService
from src.app.infra.utils.logger import get_logger

logger = get_logger(__name__)

class PoliticalElements(TypedDict):
    time: str = ""
    location: str = ""
    event: str = ""
    goal: str = ""
    subjects: List[str] = []


def preprocess_node(state: AgentState) -> AgentState:
    """
    预处理节点：文本清洗、脱敏、分类

    复用现有的 GovRequestClassifier 进行诉求分类
    """
    query = state["original_query"]
    logger.info(f"[Preprocess] 开始预处理: {query[:50]}...")

    # 1. 文本清洗和脱敏
    cleaned_query = clean_text(query)
    state["cleaned_query"] = cleaned_query

    # 2. 诉求分类（复用现有分类器）
    classifier = GovRequestClassifier()
    classification_result = classifier.classify(cleaned_query)

    state["classification"] = {
        "request_type": classification_result.request_type.value,
        "request_urgency": classification_result.request_urgency.value,
    }

    state["status"] = ProcessStatus.PREPROCESSED
    logger.info(f"[Preprocess] 分类结果: {state['classification']}")

    return state


def extract_elements_node(state: AgentState) -> AgentState:
    """
    要素提取节点：提取五大核心要素

    使用 LLM 提取：
    - 时间 (time)
    - 地点 (location)
    - 核心事件 (event)
    - 诉求目标 (goal)
    - 涉及主体 (subjects)
    """
    query = state["cleaned_query"]
    logger.info(f"[ExtractElements] 开始提取要素...")

    llm_service = create_llm_service()

    system_prompt = """你是政务问政要素提取专家。请从以下市民诉求中提取五大核心要素。

## 五大核心要素
1. 时间 (time): 事件发生的时间
2. 地点 (location): 事件发生的地点
3. 核心事件 (event): 发生了什么具体事件
4. 诉求目标 (goal): 市民希望政府做什么
5. 涉及主体 (subjects): 涉及哪些部门或人员

## 输出要求
请严格按照以下 JSON 格式输出，只输出 JSON，不要有其他内容：
{
    "time": "时间描述",
    "location": "地点描述",
    "event": "事件描述",
    "goal": "诉求目标",
    "subjects": ["主体1", "主体2"]
}
"""

    try:

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        elements = llm_service.generate_structured(
            messages,
            response_model=PoliticalElements,
            temperature=0,
            max_tokens=500,
        )

        state["political_elements"] = elements.model_dump()
        logger.info(f"[ExtractElements] 要素提取完成: {state['political_elements']}")

    except Exception as e:
        logger.warning(f"[ExtractElements] 要素提取失败: {e}")
        state["political_elements"] = {
            "time": "",
            "location": "",
            "event": "",
            "goal": "",
            "subjects": [],
        }

    return state


def clean_text(text: str) -> str:
    """
    文本清洗和脱敏
    """
    text = re.sub(r'\s+', ' ', text).strip()

    # 1. 脱敏身份证（18位，前3后4，中间11位隐藏）
    text = re.sub(r'(?<!\d)(\d{3})\d{11}(\d{4})(?!\d)', r'\1***********\2', text)

    # 2. 脱敏手机号（11位，以1开头，第二位为3-9，前后都不是数字）
    text = re.sub(r'(?<!\d)(1[3-9]\d)\d{4}(\d{4})(?!\d)', r'\1****\2', text)

    return text