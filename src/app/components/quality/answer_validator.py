"""
回答质量验证器
使用 LLM 校验回答质量的多个维度
"""

import traceback
from typing import Dict, Any
from src.app.components.quality.base_validator import BaseValidator
from src.app.infra.llm import BaseLLMService, create_llm_service
from src.app.infra.utils.logger import get_logger

logger = get_logger(__name__)


class GovAnswerValidator(BaseValidator):
    """
    回答质量验证器

    使用轻量模型校验回答质量的多个维度：
    - 相关性：回答是否与用户查询直接相关
    - 准确性：是否仅基于提供的上下文信息，无编造内容
    - 来源标注：是否注明信息来源部门和时间
    - 合规性：是否符合政务沟通规范，无敏感信息
    """

    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(GovAnswerValidator, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        # 使用轻量模型进行回答质量的校验
        self.llm_service: BaseLLMService = create_llm_service(
            provider_id="deepseek", model_name="deepseek-chat"
        )

        logger.info("✅ AnswerValidator 初始化完成")

    async def validate(
        self,
        answer: str,
        query: str,
        context: str
    ) -> Dict[str, Any]:
        """
        验证回答质量

        Args:
            answer: 生成的回答
            query: 用户查询
            context: 检索上下文

        Returns:
            质量校验结果
        """
        # 构建校验 Prompt
        validate_prompt = f"""你是回答质量校验专家，请对以下政务回复进行质量评估。

## 评估维度
1. 相关性：回答是否与用户查询直接相关
2. 准确性：是否仅基于提供的上下文信息，无编造内容
3. 来源标注：是否注明信息来源部门和时间
4. 合规性：是否符合政务沟通规范，无敏感信息

## 用户查询
{query}

## 上下文信息
{context[:1000]}

## 待评估回复
{answer}

## 输出要求
请严格按照以下 JSON 格式输出，只输出 JSON，不要有其他内容：
{{
    "relevance_score": 0.85,
    "accuracy_score": 0.9,
    "attribution_score": 0.7,
    "compliance_score": 1.0,
    "overall_score": 0.85,
    "suggestion": "优化建议（如有）"
}}
"""
        try:
            messages = [
                {"role": "system", "content": "你是一个政务回复质量评估专家"},
                {"role": "user", "content": validate_prompt}
            ]

            from pydantic import BaseModel

            class QualityScore(BaseModel):
                relevance_score: float
                accuracy_score: float
                attribution_score: float
                compliance_score: float
                overall_score: float
                suggestion: str = ""

            result = self.llm_service.generate_structured(
                messages,
                response_model=QualityScore,
                temperature=0,
                max_tokens=500,
            )

            return result.model_dump()

        except Exception as e:
            logger.error(f"❌ 回答质量校验失败: {e}")
            logger.error(traceback.format_exc())
            return {
                "relevance_score": 0.5,
                "accuracy_score": 0.5,
                "attribution_score": 0.5,
                "compliance_score": 0.5,
                "overall_score": 0.5,
                "suggestion": f"校验失败：{str(e)}"
            }