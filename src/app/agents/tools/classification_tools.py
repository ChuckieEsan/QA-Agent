"""
分类工具函数 - 政务诉求分类

封装 GovClassifier 组件，提供问政类型分类能力
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

from src.app.components.classifier import GovClassifier


class ClassifyInput(BaseModel):
    """分类工具输入 schema"""
    query: str = Field(description="用户诉求文本")


@tool(args_schema=ClassifyInput)
def classify(query: str) -> Dict[str, Any]:
    """
    分类问政类型（建议/投诉/求助/咨询）

    Args:
        query: 用户诉求文本

    Returns:
        {
            "type": str,           # 问政类型
            "confidence": float,   # 置信度
            "reason": str,         # 判定理由
            "urgency_level": str,  # 紧急程度
            "department": str      # 建议办理部门
        }
    """
    try:
        classifier = GovClassifier()
        import asyncio

        # 在同步函数中运行异步方法
        result = asyncio.run(classifier.classify_gov_request(query))
        return result
    except Exception as e:
        return {
            "type": "未知",
            "confidence": 0.0,
            "reason": f"分类失败：{str(e)}",
            "urgency_level": "一般",
            "department": "未知"
        }


# 导出所有工具
__all__ = ["classify"]
