"""
检索器基类 - LangChain BaseRetriever 实现

继承 langchain_core.retrievers.BaseRetriever，符合 LangChain 最佳实践
"""

from abc import abstractmethod
from typing import List, Dict, Any, Optional

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import Field


class LangChainRetriever(BaseRetriever):
    """
    LangChain 兼容的检索器基类

    继承自 langchain_core BaseRetriever，支持：
    - LCEL 链式调用
    - 同步/异步检索
    - LangChain 生态集成
    """

    # 配置字段
    top_k: int = Field(default=5, description="返回结果数量")
    min_similarity: float = Field(default=0.5, description="最小相似度阈值")
    cache_enabled: bool = Field(default=True, description="是否启用缓存")

    @abstractmethod
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        同步检索文档

        Args:
            query: 查询文本

        Returns:
            List[Document]: 检索到的文档列表
        """
        pass

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """
        异步检索文档（默认调用同步版本）

        Args:
            query: 查询文本

        Returns:
            List[Document]: 检索到的文档列表
        """
        return self._get_relevant_documents(query)