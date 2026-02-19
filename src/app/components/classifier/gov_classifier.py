"""
政务问政分类器
基于 LLM 的问政请求分类实现
"""

from typing import Dict, List
from src.app.components.classifier.base_classifier import BaseClassifier, GovRequestType
from src.app.infra.llm.llm_service import LLMService, get_llm_service
from src.app.infra.utils.logger import get_logger

logger = get_logger(__name__)


class GovClassifier(BaseClassifier):
    """
    政务问政分类器

    使用 LLM 对市民诉求进行分类
    """

    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(GovClassifier, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        self.llm_service: LLMService = get_llm_service()
        self._initialized = True
        logger.info("✅ Gov Classifier 初始化完成")

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

        # 调用 LLM 进行分类
        result = await self.llm_service.generate_response(
            query="",
            context=classification_prompt,
            history=None,
            stream=False
        )

        try:
            import json
            classification = json.loads(result["answer"])

            # 验证类型
            if classification["type"] not in ["advice", "complaint", "help", "consult"]:
                classification["type"] = "consult"
                classification["confidence"] = 0.5
                classification["reasoning"] = "类型识别失败，默认为咨询"

            return classification
        except Exception as e:
            # 默认分类为咨询
            logger.warning(f"⚠️  分类失败: {e}，使用默认分类")
            return {
                "type": "consult",
                "confidence": 0.5,
                "keywords": [],
                "reasoning": "无法准确分类，默认为咨询"
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
