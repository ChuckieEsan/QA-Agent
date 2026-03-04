"""
生成工具函数 - 回答生成

封装 LLMGenerator 组件，提供文本生成能力
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

from src.app.components.generators import LLMGenerator


class GenerateAnswerInput(BaseModel):
    """生成工具输入 schema"""
    query: str = Field(description="用户查询")
    context: str = Field(default="", description="检索到的上下文信息")


@tool(args_schema=GenerateAnswerInput)
def generate_answer(
    query: str,
    context: str = ""
) -> Dict[str, Any]:
    """
    基于检索结果生成政务问答回复

    Args:
        query: 用户查询
        context: 检索到的上下文信息（可选）

    Returns:
        {
            "answer": str,      # 生成的回答文本
            "metadata": Dict    # 元数据
        }
    """
    try:
        generator = LLMGenerator()
        import asyncio

        # 构建完整的 prompt
        if context:
            prompt = f"基于以下上下文回答问题：\n\n{context}\n\n问题：{query}"
        else:
            prompt = query

        answer = asyncio.run(generator.generate(prompt=prompt))

        return {
            "answer": answer,
            "metadata": {"length": len(answer)}
        }
    except Exception as e:
        return {
            "answer": f"抱歉，生成回答时出现错误：{str(e)}",
            "metadata": {"error": str(e)}
        }


# 导出所有工具
__all__ = ["generate_answer"]
