"""
验证工具函数 - 回答质量验证

封装 AnswerValidator 组件，提供回答质量验证能力
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

from src.app.components.quality import AnswerValidator


class ValidateAnswerInput(BaseModel):
    """验证工具输入 schema"""
    answer: str = Field(description="生成的回答文本")
    query: str = Field(description="用户查询")
    context: str = Field(default="", description="上下文信息")


@tool(args_schema=ValidateAnswerInput)
def validate_answer(
    answer: str,
    query: str,
    context: str = ""
) -> Dict[str, Any]:
    """
    验证回答是否符合网络问政场景的规范

    Args:
        answer: 生成的回答文本
        query: 用户查询
        context: 上下文信息（可选）

    Returns:
        {
            "overall_score": float,       # 综合评分（0-1）
            "relevance_score": float,     # 相关性评分
            "completeness_score": float,  # 完整性评分
            "accuracy_score": float,      # 准确性评分
            "passed": bool,               # 是否通过验证
            "feedback": str               # 反馈信息
        }
    """
    try:
        validator = AnswerValidator()
        import asyncio

        validation = asyncio.run(validator.validate(answer, query, context))

        return validation
    except Exception as e:
        return {
            "overall_score": 0.0,
            "relevance_score": 0.0,
            "completeness_score": 0.0,
            "accuracy_score": 0.0,
            "passed": False,
            "feedback": f"验证失败：{str(e)}"
        }


# 导出所有工具
__all__ = ["validate_answer"]
