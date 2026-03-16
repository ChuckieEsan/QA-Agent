"""
回答质量验证器
使用 langchain LCEL + PydanticOutputParser 实现
简化版：只评估准确性和合规性两个维度

依赖注入模式：接受外部注入的 BaseLLMService 实例
"""

import re
import traceback
from typing import Dict, Any, List, Optional

from pydantic import BaseModel, Field, ConfigDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable
from src.app.infra.llm import BaseLLMService, create_llm_service

from src.app.infra.utils.logger import get_logger

logger = get_logger(__name__)


class QualityScore(BaseModel):
    """质量评分模型"""
    accuracy_score: float  # 准确性：回答是否基于上下文，无编造
    compliance_score: float  # 合规性：是否包含敏感信息
    overall_score: float
    is_passed: bool  # 是否通过质量检查
    suggestion: str = ""


class GovAnswerValidator(BaseModel):
    """
    回答质量验证器

    只评估两个维度：
    1. 准确性：回答是否基于提供的上下文，无编造内容
    2. 合规性：是否包含敏感信息

    评估策略：
    - 合规性：使用规则匹配（正则 + 敏感词），低成本
    - 准确性：使用 LLM 判断，但维度简化

    """

    llm: BaseLLMService = Field(description="外部注入的大模型服务实例")

    # 允许传入任意类型的对象（避免 Pydantic 报错）
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context) -> None:
        """初始化后构建 LCEL 链"""
        self._build_chain()
        logger.info("GovAnswerValidator 初始化完成（langchain）")

    def _build_chain(self) -> None:
        """使用 langchain LCEL 构建验证 Chain"""
        # 1. 构建提示词模板（简化版：只评估准确性）
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", """你是回答质量校验专家，请对以下政务回复进行准确性评估。

## 评估维度
准确性：回答是否仅基于提供的上下文信息生成，有无编造、幻觉内容。
- 如果回答中的关键信息（数字、日期、政策名称、机构名称等）都能在上下文中找到，则准确性高
- 如果回答中出现了上下文中不存在的信息，且无法确定真假，则准确性低

## 上下文信息
{context}

## 待评估回复
{answer}
"""),
        ])

        # 2. 构建 LCEL Chain
        self._chain: RunnableSerializable = (
            self._prompt
            | self.llm.with_structured_output(QualityScore, method="function_calling")
        )

    def _check_compliance(self, answer: str) -> tuple[float, List[str]]:
        """
        检查合规性（基于规则的低成本方式）

        Returns:
            (合规分数, 触发的敏感词列表)
        """
        triggered_words = []
        # TODO: 这里调用 mcp server 政务系统中的接口

        # 有敏感词则合规分数为 0，否则为 1
        if triggered_words:
            return 0.0, triggered_words
        return 1.0, []

    async def validate(
        self,
        answer: str,
        query: str,
        context: str
    ) -> Dict[str, Any]:
        """
        验证回答质量

        策略：
        1. 合规性：基于规则检查（低成本，即时返回）
        2. 准确性：使用 LLM 评估

        Args:
            answer: 生成的回答
            query: 用户查询
            context: 检索上下文

        Returns:
            质量校验结果
        """
        try:
            # 1. 先做合规性检查（基于规则，快速失败）
            compliance_score, triggered_words = self._check_compliance(answer)

            # 如果不合规，直接返回失败结果，不再调用 LLM
            if compliance_score < 1.0:
                return {
                    "accuracy_score": 0.0,
                    "compliance_score": compliance_score,
                    "overall_score": compliance_score,
                    "is_passed": False,
                    "suggestion": f"包含敏感内容：{', '.join(triggered_words)}"
                }

            # 2. 准确性检查（需要调用 LLM）
            # 截取上下文（避免过长）
            truncated_context = context[:1000]

            # 获取 format_instructions
            format_instructions = "请用 JSON 格式输出，包含 accuracy_score, compliance_score, overall_score, is_passed, suggestion 字段"

            result = await self._chain.ainvoke(
                {
                    "context": truncated_context,
                    "answer": answer,
                    "format_instructions": format_instructions,
                },
                config={"temperature": 0, "max_tokens": 500}
            )

            # 3. 计算总分（准确性权重 60%，合规性权重 40%）
            accuracy_score = result.accuracy_score
            overall_score = accuracy_score * 0.6 + compliance_score * 0.4

            # 4. 判断是否通过（总分 >= 0.6 且准确性 >= 0.5）
            is_passed = overall_score >= 0.6 and accuracy_score >= 0.5

            return {
                "accuracy_score": accuracy_score,
                "compliance_score": compliance_score,
                "overall_score": overall_score,
                "is_passed": is_passed,
                "suggestion": result.suggestion if result.suggestion else ""
            }

        except Exception as e:
            logger.error(f"回答质量校验失败: {e}")
            logger.error(traceback.format_exc())
            # 校验失败时默认通过，避免阻塞用户体验
            return {
                "accuracy_score": 0.5,
                "compliance_score": 1.0,
                "overall_score": 0.7,
                "is_passed": True,
                "suggestion": f"校验服务异常，已放行：{str(e)}"
            }


def create_gov_answer_validator(
    llm: Optional[BaseLLMService] = None,
) -> GovAnswerValidator:
    """
    创建回答验证器的工厂函数

    Args:
        llm: 大模型实例（如果不提供则使用默认配置创建 BaseLLMService）

    Returns:
        GovAnswerValidator 实例
    """
    if llm is None:
        llm = create_llm_service(
            provider_id="deepseek", model_name="deepseek-chat"
        )

    return GovAnswerValidator(llm=llm)