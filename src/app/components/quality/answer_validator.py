"""
回答质量验证器
使用 LLM 校验回答质量的多个维度
"""

import json
import traceback
from typing import Dict, Any
from dashscope import Generation
from src.app.components.quality.base_validator import BaseValidator
from src.app.infra.llm.multi_model_service import get_light_llm_service
from src.app.infra.utils.logger import get_logger

logger = get_logger(__name__)


class AnswerValidator(BaseValidator):
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
            cls._instance = super(AnswerValidator, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        self._initialized = True
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
        validate_prompt = f"""
你是回答质量校验 Agent，请校验以下回答是否符合要求：

## 校验标准
1. 相关性：回答是否与用户查询 ({query}) 直接相关
2. 准确性：是否仅基于提供的上下文信息，无编造内容
3. 来源标注：是否注明信息来源部门和时间
4. 合规性：是否符合政务沟通规范，无敏感信息

## 待校验内容
上下文：{context[:1000]}...
回答：{answer}

## 输出格式（JSON）
{{
    "relevance_score": 0.0-1.0,
    "accuracy_score": 0.0-1.0,
    "attribution_score": 0.0-1.0,
    "compliance_score": 0.0-1.0,
    "overall_score": 0.0-1.0,
    "suggestion": "优化建议（可选）"
}}
"""

        try:
            # 使用轻量模型进行质量校验（降低成本）
            light_llm = get_light_llm_service()
            response = Generation.call(
                model=light_llm.get_model_name(),
                prompt=validate_prompt,
                temperature=light_llm.get_config().temperature,
                max_tokens=300,
                result_format="text"
            )

            if response.status_code == 200:
                validate_result = json.loads(response.output.text)
                return validate_result
            else:
                raise Exception(f"质量校验失败: {response.code}")

        except Exception as e:
            logger.error(f"❌ 回答质量校验失败: {e}")
            logger.error(traceback.format_exc())
            return {
                "relevance_score": 0.0,
                "accuracy_score": 0.0,
                "attribution_score": 0.0,
                "compliance_score": 0.0,
                "overall_score": 0.0,
                "suggestion": f"校验失败：{str(e)}"
            }

    async def initialize(self) -> None:
        """初始化验证器资源"""
        pass
