"""
行政权力清单检索器 - LangChain 最佳实践实现

继承 LangChain BaseRetriever，返回标准 Document 格式
依赖注入模式：通过字段接收外部注入的 Embedding 模型和 Milvus 客户端
"""

from typing import List, Optional

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import Field, ConfigDict

from src.config.setting import settings
from src.app.infra.embedding.base_embedding import BaseEmbedding
from src.app.infra.db.milvus_db import MilvusDBClient
from src.app.infra.utils.logger import get_logger

logger = get_logger(__name__)


class PowersVectorRetriever(BaseRetriever):
    """
    行政权力清单向量检索器

    继承 LangChainRetriever，专门用于检索行政权力清单数据
    依赖注入模式：通过字段接收外部注入的依赖实例
    """

    # 依赖注入字段
    embed_model: BaseEmbedding = Field(description="Embedding 模型实例")
    milvus_client: MilvusDBClient = Field(description="Milvus 数据库客户端实例")

    # 配置字段
    top_k: int = Field(default=5, description="返回结果数量")
    min_similarity: float = Field(default=0.5, description="最小相似度阈值")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        执行行政权力清单检索

        Args:
            query: 查询文本

        Returns:
            List[Document]: 检索到的文档列表
        """
        # 向量检索
        query_vec = self.embed_model.model.encode([query], normalize_embeddings=True)

        search_limit = self.top_k * 3
        raw_results = self.milvus_client.search(
            collection_name=settings.vectordb.gov_powers_collection_name,
            vectors=query_vec.tolist(),
            top_k=search_limit,
            output_fields=["text", "department", "power_type", "power_name", "doc_type"],
        )

        if not raw_results or not raw_results[0]:
            return []

        # 转换结果
        documents = self._process_results(raw_results[0])

        # 阈值筛选
        documents = self._threshold_filter(documents)

        # 截取最终结果
        return documents[:self.top_k]

    def _process_results(self, raw_hits: list) -> List[Document]:
        """将原始结果转换为 Document"""
        documents = []
        for hit in raw_hits:
            entity = hit.get("entity", hit)

            documents.append(
                Document(
                    page_content=entity.get("text", ""),
                    metadata={
                        "source": "milvus",
                        "collection": settings.vectordb.gov_powers_collection_name,
                        "department": entity.get("department", ""),
                        "power_type": entity.get("power_type", ""),
                        "power_name": entity.get("power_name", ""),
                        "similarity": 1 - hit.get("distance", 0),
                    }
                )
            )
        return documents

    def _threshold_filter(self, documents: List[Document]) -> List[Document]:
        """阈值筛选"""
        return [
            doc for doc in documents
            if doc.metadata.get("similarity", 0) >= self.min_similarity
        ]


def create_powers_retriever(
    embed_model: Optional[BaseEmbedding] = None,
    milvus_client: Optional[MilvusDBClient] = None,
    top_k: int = 5,
    min_similarity: float = 0.5,
) -> PowersVectorRetriever:
    """
    创建行政权力清单检索器的工厂函数

    Args:
        embed_model: Embedding 模型实例（如果不提供则使用单例）
        milvus_client: Milvus 客户端实例（如果不提供则使用单例）
        top_k: 返回结果数量
        min_similarity: 最小相似度阈值

    Returns:
        PowersVectorRetriever 实例
    """
    if embed_model is None:
        embed_model = BaseEmbedding()
    if milvus_client is None:
        milvus_client = MilvusDBClient()

    return PowersVectorRetriever(
        embed_model=embed_model,
        milvus_client=milvus_client,
        top_k=top_k or settings.vectordb.default_top_k,
        min_similarity=min_similarity or settings.retriever.min_similarity,
    )