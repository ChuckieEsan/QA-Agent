"""
政务问政分类器
基于 LLM 的问政请求分类实现
"""

import json
import traceback
from typing import Dict, List
from src.app.components.classifier.base_classifier import BaseClassifier, GovRequestType
from src.app.infra.llm.multi_model_service import get_light_llm_service, LLMService
from src.app.infra.utils.logger import get_logger
from dashscope import Generation

logger = get_logger(__name__)


class GovClassifier(BaseClassifier):
    """
    政务问政分类器

    使用轻量级 LLM 对市民诉求进行分类（优化 token 消耗）
    """

    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(GovClassifier, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        # 使用轻量模型进行分类（降低成本）
        self.llm_service: LLMService = get_light_llm_service()
        self._initialized = True
        logger.info(f"✅ Gov Classifier 初始化完成（使用轻量模型: {self.llm_service.get_model_name()}）")

    async def classify_gov_request(self, text: str, **kwargs) -> Dict[str, any]:
        """
        分类问政请求

        Args:
            text: 市民诉求文本
            **kwargs: 其他分类参数

        Returns:
            {
                "type": "advice|complaint|help|consult",  # 问政类型
                "confidence": float,                       # 置信度 (0-1)
                "keywords": List[str],                     # 关键词
                "reasoning": str                           # 分类理由
            }
        """
        classification_prompt = f"""
        你是政务问政分类专家，请对以下市民诉求进行分类：

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

        ## 市民诉求
        {text}

        ## 输出格式（JSON）
        {{
            "type": "advice|complaint|help|consult",
            "confidence": 0.0-1.0,
            "keywords": ["关键词1", "关键词2"],
            "reasoning": "分类理由（50字以内）"
        }}
        """

        # 使用轻量模型进行分类（降低成本，提高速度）
        try:
            response = Generation.call(
                model=self.llm_service.get_model_name(),
                prompt=classification_prompt,
                temperature=self.llm_service.get_config().temperature,
                max_tokens=self.llm_service.get_config().max_tokens,
                top_p=self.llm_service.get_config().top_p,
                result_format='text'  # 注意：使用 text 格式
            )

            if response.status_code == 200:
                # 注意：result_format='text' 时，结果在 output.text 而不是 choices
                classification_str = response.output.text
                classification = json.loads(classification_str)

                # 验证类型
                if classification["type"] not in ["advice", "complaint", "help", "consult"]:
                    classification["type"] = "consult"
                    classification["confidence"] = 0.5
                    classification["reasoning"] = "类型识别失败，默认为咨询"

                return classification
            else:
                raise Exception(f"API调用失败: {response.code} - {response.message}")

        except Exception as e:
            # 默认分类为咨询
            logger.warning(f"⚠️  分类失败: {e}，使用默认分类")
            logger.warning(traceback.format_exc())
            return {
                "type": "consult",
                "confidence": 0.5,
                "keywords": [],
                "reasoning": f"无法准确分类，错误: {str(e)}"
            }

    async def classify_batch(self, texts: List[str], **kwargs) -> List[Dict[str, any]]:
        """
        批量分类问政请求

        Args:
            texts: 市民诉求文本列表
            **kwargs: 其他分类参数

        Returns:
            分类结果列表
        """
        results = []
        for text in texts:
            result = await self.classify_gov_request(text)
            results.append(result)
        return results

    async def initialize(self) -> None:
        """初始化分类器资源"""
        # 预热 LLM Service
        await self.classify_gov_request("测试分类")
        logger.info("✅ Gov Classifier 预热完成")
