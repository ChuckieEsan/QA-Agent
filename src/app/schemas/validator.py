"""验证领域数据模型"""

from pydantic import BaseModel, Field


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