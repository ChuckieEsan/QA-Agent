"""
政务问政分类器
基于 LLM Service 的 JSON 输出实现
"""

import json
import traceback
from typing import Dict, List, Any, TypedDict
from pydantic import BaseModel
from src.app.components.classifier.base_classifier import (
    BaseClassifier,
    GovRequestClassifiedResult,
)
from src.app.infra.llm.base_llm_service import BaseLLMService
from src.app.infra.llm import create_llm_service
from src.app.infra.utils.logger import get_logger

logger = get_logger(__name__)


class GovRequestClassifier(BaseClassifier):
    """
    政务问政分类器
    """

    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(GovRequestClassifier, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        # 使用轻量模型进行分类
        self.llm_service: BaseLLMService = create_llm_service(
            provider_id="deepseek", model_name="deepseek-chat"
        )
        self._initialized = True
        logger.info(
            f"RequestClassifier 初始化完成（使用轻量模型: {self.llm_service._model_name}）"
        )

    def classify(self, text: str) -> GovRequestClassifiedResult:
        """
        分类问政请求

        Args:
            text: 市民诉求文本
            **kwargs: 其他分类参数

        Returns:
            GovRequestType
        """
        # 构建消息
        messages = self._build_classification_prompt(text)

        # 使用 LLM Service 的 JSON 输出
        try:
            result = self.llm_service.generate_structured(
                messages,
                response_model=GovRequestClassifiedResult,
                temperature=0,
                max_tokens=500,
            )

            return result

        except Exception as e:
            logger.warning(f"分类失败: {e}，使用默认分类")
            logger.warning(traceback.format_exc())
            return {"request_type": "consult", "request_urgency": "normal"}

    def _build_classification_prompt(self, text: str) -> List[Dict[str, Any]]:
        """构建分类提示词"""
        system_prompt = """你是政务问政分类专家，请对以下市民诉求进行分类，并分析问政请求的紧急情况。

## 分类标准
1. 建议（advice）：对政府工作提出改进建议、意见
   - 关键词：建议、希望、改进、优化、提升
   - 示例："建议增加公交车班次"

2. 投诉（complaint）：反映政府部门或工作人员的问题、不当行为
   - 关键词：投诉、不满、违规、问题、差
   - 示例："投诉某部门办事效率低"

3. 求助（help）：请求政府帮助解决个人或家庭困难
   - 关键词：求助、帮忙、解决、困难、申请
   - 示例："我家房子漏水，请求帮助"

4. 咨询（consult）：询问政策、流程、办事指南等信息
   - 关键词：咨询、请问、如何、怎么、什么
   - 示例："咨询雨露计划申请条件"

5. 其他（other）：与政务无关的内容
   - 示例：1 + 1 等于几？今天天气怎么样

## 输出要求
你只能按照以下 JSON Schema 输出，包含以下字段.
- request_type: 分类类型（advice/complaint/help/consult/other）
- request_urgency: 紧急程度（major/normal/minor）
"""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]
