"""
检索工具函数 - 知识库检索

封装 HybridVectorRetriever 组件，提供向量检索能力
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

from src.app.components.retrievers import HybridVectorRetriever


class RetrieveInput(BaseModel):
    """检索工具输入 schema"""
    query: str = Field(description="检索查询语句")
    top_k: int = Field(default=5, description="返回结果数量")
    threshold: float = Field(default=0.5, description="相似度阈值")


@tool(args_schema=RetrieveInput)
def retrieve(
    query: str,
    top_k: int = 5,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    检索知识库已有的网络问政相关案例

    Args:
        query: 检索查询语句
        top_k: 返回结果数量（默认 5）
        threshold: 相似度阈值（默认 0.5）

    Returns:
        {
            "context": str,          # 检索到的上下文文本
            "results": List[Dict],   # 检索结果列表
            "metadata": Dict         # 元数据（检索耗时等）
        }
    """
    try:
        retriever = HybridVectorRetriever()
        context, results, metadata = retriever.retrieve(
            query=query,
            top_k=top_k,
            threshold=threshold
        )

        return {
            "context": context,
            "results": results,
            "metadata": metadata
        }
    except Exception as e:
        return {
            "context": "",
            "results": [],
            "metadata": {"error": str(e)}
        }


# 导出所有工具
__all__ = ["retrieve"]
