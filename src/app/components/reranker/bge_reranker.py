"""
BGE 文档压缩器 - LangChain 组件层

继承 langchain_core.document_compressors.BaseDocumentCompressor
每次请求实例化，通过依赖注入获取 bge_client
"""

from typing import List, Optional

from langchain_core.documents import BaseDocumentCompressor
from langchain_core.documents import Document
from pydantic import Field

from src.app.infra.reranker.base_reranker import BaseRerankerClient
from src.app.infra.utils.logger import get_logger

logger = get_logger(__name__)


class BGERerankerCompressor(BaseDocumentCompressor):
    """
    BGE 文档压缩器

    继承 BaseDocumentCompressor，用于 ContextualCompressionRetriever
    每次请求实例化，不是单例
    """

    # 通过依赖注入接收 bge_client
    bge_client: BaseRerankerClient = Field(description="BGE Reranker 客户端")

    # 可选配置
    top_k: int = Field(default=5, description="返回前 K 个结果")
    min_score: float = Field(default=0.0, description="最小得分阈值")

    class Config:
        """Pydantic 配置"""
        arbitrary_types_allowed = True

    def compress_documents(
        self,
        documents: List[Document],
        query: str,
    ) -> List[Document]:
        """
        对文档进行重排和过滤

        Args:
            documents: 待压缩的文档列表
            query: 查询文本

        Returns:
            重排后的文档列表
        """
        if not documents:
            return []

        # 如果只有一个文档，直接返回
        if len(documents) == 1:
            return documents

        # 提取文本内容
        texts = [doc.page_content for doc in documents]

        # 计算相关性得分
        scores = self.bge_client.compute_score(query, texts)

        # 为每个文档添加重排得分
        reranked_docs = []
        for doc, score in zip(documents, scores):
            # 复制文档避免修改原始数据
            new_doc = Document(
                page_content=doc.page_content,
                metadata=dict(doc.metadata)
            )
            new_doc.metadata["rerank_score"] = score
            new_doc.metadata["composite_score"] = score
            reranked_docs.append(new_doc)

        # 按重排得分降序排序
        reranked_docs.sort(
            key=lambda x: x.metadata.get("rerank_score", 0),
            reverse=True
        )

        # 过滤低分文档
        reranked_docs = [
            doc for doc in reranked_docs
            if doc.metadata.get("rerank_score", 0) >= self.min_score
        ]

        # 返回前 top_k 个结果
        return reranked_docs[:self.top_k]

    def compress_query(self, query: str) -> str:
        """
        压缩查询（可选实现）

        Args:
            query: 原始查询

        Returns:
            压缩后的查询（这里直接返回原始查询）
        """
        return query


def create_bge_compressor(
    bge_client: Optional[BaseRerankerClient] = None,
    top_k: int = 5,
    min_score: float = 0.0,
) -> BGERerankerCompressor:
    """
    创建 BGE 压缩器的工厂函数

    Args:
        bge_client: BGE 客户端（如果不提供则自动获取单例）
        top_k: 返回前 K 个结果
        min_score: 最小得分阈值

    Returns:
        BGECompressor 实例
    """
    if bge_client is None:
        bge_client = BaseRerankerClient()

    return BGERerankerCompressor(
        bge_client=bge_client,
        top_k=top_k,
        min_score=min_score,
    )