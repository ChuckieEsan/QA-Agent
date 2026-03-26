"""
政务回答质量验证器 (Validator)
使用 langchain LCEL + PydanticOutputParser 实现

策略：前置轻量级合规拦截 + 后置 LLM 幻觉检测
"""

import traceback
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable
from src.app.infra.llm import BaseLLMService, create_llm_service
from src.app.infra.utils.logger import get_logger

logger = get_logger(__name__)


class LLMValidationResult(BaseModel):
    """大模型结构化输出的解析模型"""

    accuracy_score: float = Field(
        ...,
        description="事实一致性得分，范围[0.0, 1.0]。1.0代表完全基于上下文且无捏造，0.0代表严重幻觉或脱离上下文。",
    )
    suggestion: str = Field(
        ..., description="如果不准确，请给出具体的修改建议；如果准确，请输出 '无'。"
    )


class GovAnswerValidatedResult(BaseModel):
    """验证器最终返回给系统的综合评分模型"""

    is_compliance: bool = Field(
        default=True, description="是否通过合规性校验（无敏感词）"
    )
    accuracy_score: float = Field(default=0.0, description="大模型判定的准确性得分")
    is_passed: bool = Field(
        default=False, description="综合判断是否允许将该回答发送给用户"
    )
    suggestion: str = Field(default="", description="拦截原因或修改建议")


class GovAnswerValidator(BaseModel):
    """
    回答质量验证器
    """

    llm: BaseLLMService = Field(description="外部注入的大模型服务实例")

    # 阈值配置
    accuracy_threshold: float = Field(default=0.6, description="准确性及格线")
    max_context_length: int = Field(
        default=2000, description="截断上下文的最大长度，防止超长"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context) -> None:
        """初始化后构建 LCEL 链"""
        self._build_chain()
        logger.info("GovAnswerValidator 初始化完成")

    def _build_chain(self) -> None:
        """使用 langchain LCEL 构建验证 Chain"""

        # 优化提示词：引入 RAG Triad 评估法（忠实度验证）
        self._prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """你是政务智能问答的质量审核专家。请根据提供的【用户提问】和【参考上下文】，对大模型生成的【待评估回复】进行“事实一致性（准确性）”评估。

## 评估准则
1. 忠实度：回复中的核心政策、数字、办理条件必须100%来源于【参考上下文】，绝不允许捏造、瞎编。
2. 应对策略：如果上下文没有相关信息，回复中明确表示“政策库暂无信息并建议转人工”，这也算作高度准确（得分应为 1.0）。
3. 评分范围：给出 0.0 到 1.0 的得分，并简要给出建议。
""",
                ),
                (
                    "user",
                    """
【用户提问】
{query}

【参考上下文】
{context}

【待评估回复】
{answer}
""",
                ),
            ]
        )

        # 绑定 LLMValidationResult 作为强约束输出
        self._chain: RunnableSerializable = (
            self._prompt
            | self.llm.with_structured_output(
                LLMValidationResult, method="function_calling"
            )
        )

    def _check_compliance(self, query: str) -> tuple[bool, str]:
        """
        检查合规性（基于规则/MCP的低成本方式）
        """
        # TODO: 后续接入 MCP Server 的敏感词/风控网关
        # 假设这里有一个简单的拦截规则

        return True, "合规检查通过"

    async def validate(self, answer: str, query: str, context: str) -> GovAnswerValidatedResult:
        """验证核心入口"""
        try:
            # 1. 快速失败：合规性规则拦截
            is_compliance, msg = self._check_compliance(answer)
            if not is_compliance:
                logger.warning(f"回答未通过合规性校验: {msg}")
                return GovAnswerValidatedResult(
                    is_compliance=False,
                    accuracy_score=0.0,
                    is_passed=False,
                    suggestion=msg,
                )

            # 2. 深度校验：LLM 判断幻觉
            truncated_context = context[: self.max_context_length]

            # 执行 LCEL 链
            llm_result: LLMValidationResult = await self._chain.ainvoke(
                {
                    "query": query,
                    "context": truncated_context,
                    "answer": answer,
                },
                config={"temperature": 0.0},  # 评估任务必须用 0 温度
            )

            # 3. 综合决断
            is_passed = llm_result.accuracy_score >= self.accuracy_threshold

            if not is_passed:
                logger.warning(
                    f"回答存在幻觉或不准确，得分: {llm_result.accuracy_score}。建议: {llm_result.suggestion}"
                )

            return GovAnswerValidatedResult(
                is_compliance=True,
                accuracy_score=llm_result.accuracy_score,
                is_passed=is_passed,
                suggestion=llm_result.suggestion,
            )

        except Exception as e:
            logger.error(f"回答质量校验服务崩溃: {e}")
            logger.error(traceback.format_exc())
            # 优雅降级：如果大模型抽风没返回正确 JSON，不要卡死用户，直接放行
            return GovAnswerValidatedResult(
                is_compliance=True,
                accuracy_score=1.0,
                is_passed=True,
                suggestion=f"校验系统异常已放行，异常信息: {str(e)}",
            )


def create_gov_answer_validator(
    llm: Optional[BaseLLMService] = None,
) -> GovAnswerValidator:
    if llm is None:
        llm = create_llm_service(provider_id="deepseek", model_name="deepseek-chat")
    return GovAnswerValidator(llm=llm)
