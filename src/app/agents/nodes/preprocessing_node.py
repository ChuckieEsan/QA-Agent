"""
预处理 Node - 诉求文本清洗和脱敏

负责：
1. 文本清洗（去除特殊符号、冗余空格等）
2. 敏感信息脱敏
3. 核心要素提取
"""

import re
from typing import Dict, Any
from src.app.agents.state import AppealState
from src.app.infra.utils.logger import get_logger

logger = get_logger(__name__)


def preprocess_query(state: AppealState) -> AppealState:
    """
    预处理节点 - 清洗和脱敏用户诉求

    Args:
        state: 当前状态

    Returns:
        更新后的状态
    """
    logger.info("[PreprocessingNode] 开始预处理诉求")

    raw_query = state.get("raw_query", "")

    if not raw_query:
        state["error_message"] = "诉作文本为空"
        state["current_step"] = "preprocessing_failed"
        return state

    try:
        # 1. 文本清洗
        cleaned = clean_text(raw_query)
        state["cleaned_query"] = cleaned
        logger.debug(f"[PreprocessingNode] 清洗后：{cleaned[:50]}...")

        # 2. 敏感信息脱敏
        desensitized = desensitize_text(cleaned)
        state["desensitized_query"] = desensitized
        logger.debug(f"[PreprocessingNode] 脱敏后：{desensitized[:50]}...")

        # 3. 核心要素提取（简化版，使用规则提取）
        elements = extract_elements(cleaned)
        state["extracted_elements"] = elements
        logger.debug(f"[PreprocessingNode] 提取要素：{elements}")

        state["current_step"] = "preprocessing_completed"
        logger.info("[PreprocessingNode] 预处理完成")

    except Exception as e:
        logger.error(f"[PreprocessingNode] 预处理失败：{e}")
        state["error_message"] = str(e)
        state["current_step"] = "preprocessing_failed"

    return state


def clean_text(text: str) -> str:
    """
    文本清洗

    - 去除特殊符号
    - 去除冗余空格
    - 规范化格式
    """
    # 去除特殊符号（保留中文标点和基本标点）
    text = re.sub(r'[^\w\s\u4e00-\u9fa5,.!?;:()""'',，。！？、；：（）\s]', '', text)

    # 去除多余空格（保留单个空格）
    text = re.sub(r'\s+', ' ', text)

    # 去除首尾空格
    text = text.strip()

    return text


def desensitize_text(text: str) -> str:
    """
    敏感信息脱敏

    脱敏类型：
    - 手机号：138****1234
    - 身份证号：5105***********1234
    - 姓名：张*（当姓名前有"姓名："或"名："标识时）
    - 详细地址：XX 小区/XX 路
    """
    # 手机号脱敏：13812341234 -> 138****1234
    text = re.sub(
        r'(\d{3})\d{4}(\d{4})',
        r'\1****\2',
        text
    )

    # 身份证号脱敏：510521199001011234 -> 5105***********1234
    text = re.sub(
        r'(\d{4})\d{10}(\d{4})',
        r'\1***********\2',
        text
    )

    # 姓名脱敏：张三 -> 张*（当姓名前有"姓名："或"名："标识时）
    # 使用简单模式，避免 look-behind 可变宽度问题
    text = re.sub(
        r'(?:姓名 | 名)[:：]?\s*([\u4e00-\u9fa5]{2,4})',
        lambda m: '姓名：' + m.group(1)[0] + '*' * (len(m.group(1)) - 1),
        text
    )

    # 详细地址脱敏
    text = re.sub(r'(\d+) 号楼 (\d+) 单元 (\d+) 号', r'\1 号楼**单元**号', text)
    text = re.sub(r'(\d+) 栋 (\d+) 楼 (\d+) 号', r'\1 栋**楼**号', text)

    return text


def extract_elements(text: str) -> Dict[str, Any]:
    """
    核心要素提取

    提取：
    - time: 时间
    - location: 地点
    - event: 事件
    - goal: 诉求目标
    """
    elements = {
        "time": "",
        "location": "",
        "event": "",
        "goal": ""
    }

    # 简单的时间提取规则
    time_patterns = [
        r'(\d{4} 年\d{1,2}月\d{1,2} 日)',
        r'(\d{4}-\d{1,2}-\d{1,2})',
        r'(\d{4}/\d{1,2}/\d{1,2})',
        r'(今天 | 昨天 | 明天 | 上周 | 本月 | 今年)',
    ]
    for pattern in time_patterns:
        match = re.search(pattern, text)
        if match:
            elements["time"] = match.group(1)
            break

    # 简单的地点提取规则
    location_patterns = [
        r'(?:在 | 位于 | 地处) ([^\s,，.。]+)',
        r'([^\s,，.。]+小区)',
        r'([^\s,，.。]+街道)',
        r'([^\s,，.。]+路)',
    ]
    for pattern in location_patterns:
        match = re.search(pattern, text)
        if match:
            elements["location"] = match.group(1)
            break

    # 简单的事件和诉求目标提取
    # 这里使用关键词匹配，实际项目中可以使用 NER 模型
    event_keywords = ['投诉', '举报', '反映', '求助', '建议', '咨询', '问题', '情况']
    for keyword in event_keywords:
        if keyword in text:
            elements["event"] = keyword
            break

    goal_keywords = ['希望', '要求', '请求', '请', '想要', '需要']
    for keyword in goal_keywords:
        if keyword in text:
            idx = text.find(keyword)
            elements["goal"] = text[idx:idx + 50]  # 提取关键词后 50 字
            break

    return elements
