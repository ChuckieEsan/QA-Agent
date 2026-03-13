"""
行政权力清单检索器

专门用于检索行政权力清单数据（gov_powers collection）
"""

import threading
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

from src.config.setting import settings
from src.app.infra.db.milvus_db import MilvusDBClient
from src.app.infra.embedding import BaseEmbedding
from .base_retriever import BaseRetriever
from src.app.infra.utils.logger import get_logger

logger = get_logger(__name__)


class PowersVectorRetriever(BaseRetriever):
    """
    行政权力清单向量检索器

    专门用于检索行政权力清单数据，不需要重排
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, config: Optional[Dict[str, Any]] = None):
        """单例模式"""
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(PowersVectorRetriever, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化行政权力清单检索器

        Args:
            config: 配置字典（可选），如果为 None 则使用默认配置
        """
        if self._initialized:
            return

        # 合并配置
        default_config = {
            "top_k": settings.vectordb.default_top_k,
            "cache_enabled": settings.retriever.enable_cache,
            "cache_ttl": settings.retriever.cache_ttl_minutes * 60,
            "max_cache_size": settings.retriever.cache_max_size,
            "min_similarity": settings.retriever.min_similarity,
        }

        if config:
            default_config.update(config)

        super().__init__(default_config)

        logger.info("[PowersRetriever] 初始化行政权力清单检索器...")
        self.initialize()
        logger.info("[PowersRetriever] 初始化完成")
        self._initialized = True

    def initialize(self) -> None:
        """
        初始化核心资源

        实现 BaseRetriever 的抽象方法
        """
        # 1. 加载 Embedding 模型（使用单例）
        logger.info("获取 Embedding 模型单例...")
        self.embedding = BaseEmbedding()
        self.embed_model = self.embedding.model

        # 2. 连接向量数据库
        logger.info(f"连接 Milvus: {settings.vectordb.db_path} ...")
        self.milvus_client = MilvusDBClient()
        self.collection_name = settings.vectordb.gov_powers_collection_name

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """
        执行行政权力清单检索

        实现 BaseRetriever 的抽象方法

        Args:
            query: 查询文本
            top_k: 返回结果数量
            **kwargs: 其他参数

        Returns:
            (context_str, results, metadata)
            - context_str: 拼接的上下文文本
            - results: 详细的结果列表
            - metadata: 元数据字典
        """
        start_time = datetime.now()

        # 使用配置的 top_k
        if top_k is None:
            top_k = self.default_top_k

        # 1. 检查缓存
        if self.cache_enabled:
            cache_key = self._get_cache_key(query, top_k, **kwargs)
            cached_result = self._check_cache(cache_key)
            if cached_result:
                context, results, metadata = cached_result
                metadata["cache_hit"] = True
                logger.debug(f"使用缓存结果: {cache_key[:30]}...")
                return context, results, metadata

        try:
            # 2. 向量检索
            query_vec = self.embed_model.encode([query], normalize_embeddings=True)

            search_limit = top_k * 3
            raw_results = self.milvus_client.search(
                collection_name=self.collection_name,
                vectors=query_vec.tolist(),
                top_k=search_limit,
                output_fields=["text", "department", "power_type", "power_name", "doc_type"],
            )

            if not raw_results:
                result = (
                    "未在权力清单中找到相关信息。",
                    [],
                    {
                        "query": query,
                        "num_results": 0,
                        "num_raw_results": 0,
                        "avg_similarity": 0.0,
                        "threshold_applied": self.min_similarity,
                        "cache_hit": False,
                        "sources": []
                    }
                )
                if self.cache_enabled:
                    self._update_cache(cache_key, *result)
                return result

            # 3. 转换结果并计算相似度
            processed_results = []
            for hit in raw_results[0]:
                entity = hit.get("entity", hit)

                processed_hit = {
                    "entity": entity,
                    "similarity": hit.get("distance", 0),
                    "department": entity.get("department", ""),
                    "power_type": entity.get("power_type", ""),
                    "power_name": entity.get("power_name", ""),
                    "text": entity.get("text", ""),
                }
                processed_results.append(processed_hit)

            # 4. 阈值筛选
            filtered_results = self._threshold_filter(processed_results)

            # 5. 截取最终结果
            final_results = filtered_results[:min(top_k, len(filtered_results))]

            # 6. 构建上下文
            context_str = self.build_context(query, final_results)

            # 7. 准备元数据
            metadata = {
                "query": query,
                "retrieval_time": (datetime.now() - start_time).total_seconds(),
                "num_results": len(final_results),
                "num_raw_results": len(processed_results),
                "avg_similarity": (
                    np.mean([r["similarity"] for r in final_results])
                    if final_results else 0
                ),
                "threshold_applied": self.min_similarity,
                "cache_hit": False,
                "sources": [
                    {
                        "rank": i + 1,
                        "power_name": r.get("power_name", "无名称"),
                        "department": r.get("department", "未知部门"),
                        "power_type": r.get("power_type", "未知类型"),
                        "similarity": r.get("similarity", 0),
                    }
                    for i, r in enumerate(final_results)
                ],
            }

            # 8. 更新缓存
            if self.cache_enabled:
                self._update_cache(cache_key, context_str, final_results, metadata)

            return context_str, final_results, metadata

        except Exception as e:
            import traceback
            logger.error(f"[PowersRetriever] 检索失败: {e}")
            logger.error(traceback.format_exc())
            return f"检索服务暂时不可用: {str(e)}", [], {
                "query": query,
                "error": str(e),
                "num_results": 0,
                "avg_similarity": 0.0,
                "threshold_applied": self.min_similarity,
                "cache_hit": False,
                "sources": []
            }

    def _threshold_filter(self, results: List[Dict]) -> List[Dict]:
        """阈值筛选"""
        if not results:
            return []

        threshold = self.min_similarity
        filtered = [r for r in results if r["similarity"] >= threshold]

        return filtered if len(filtered) > 0 else results

    # ==================== 静态工厂方法 ====================

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PowersVectorRetriever":
        """
        从配置创建实例

        Args:
            config: 配置字典

        Returns:
            PowersVectorRetriever 实例
        """
        return cls(config=config)
    
    