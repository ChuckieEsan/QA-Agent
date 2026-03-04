"""
混合向量检索器
结合向量检索 + 多维度重排 + 缓存的完整实现
"""

import threading
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from sentence_transformers import SentenceTransformer

from src.config.setting import settings
from src.app.infra.utils import get_device
from src.app.infra.db.milvus_db import MilvusDBClient
from src.app.components.rerankers import BGEReranker
from .base_retriever import BaseRetriever
from src.app.infra.utils.logger import get_logger

logger = get_logger(__name__)


class HybridVectorRetriever(BaseRetriever):
    """
    混合向量检索器

    继承自 BaseRetriever，实现具体的向量检索逻辑
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, config: Optional[Dict[str, Any]] = None):
        """单例模式"""
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(HybridVectorRetriever, cls).__new__(cls)
        return cls._instance

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化混合检索器

        Args:
            config: 配置字典（可选），如果为 None 则使用默认配置
        """
        if getattr(self, "_is_initialized", False):
            return

        # 合并配置：用户配置优先，否则使用 settings
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

        logger.info("[HybridRetriever] 初始化混合策略检索器...")
        self.initialize()
        self._is_initialized = True
        logger.info("[HybridRetriever] 初始化完成")

    def initialize(self) -> None:
        """
        初始化核心资源

        实现 BaseRetriever 的抽象方法
        """
        # 1. 加载 Embedding 模型
        self.device = get_device()
        logger.info(f"加载 Embedding 模型: {settings.models.embedding_model_path} ...")
        self.embed_model = SentenceTransformer(
            str(settings.models.embedding_model_path),
            device=self.device
        )

        # 2. 连接向量数据库
        logger.info(f"连接 Milvus: {settings.vectordb.db_path} ...")
        self.milvus_client = MilvusDBClient()
        self.collection_name = settings.vectordb.collection_name

        # 3. 混合策略配置
        self.min_results = settings.retriever.min_results
        self.max_results = settings.retriever.max_results

        # 4. 重排权重配置
        self.rerank_weights = {
            "similarity": settings.retriever.weight_similarity,
            "recency": settings.retriever.weight_recency,
            "length": settings.retriever.weight_length,
        }

        # 5. 初始化 BGE 重排模型（如果配置了模型路径）
        self.reranker = None
        if (settings.models.reranker_model and
            settings.models.reranker_model_path and
            settings.models.reranker_model_path.exists()):
            try:
                logger.info(f"加载 BGE 重排模型: {settings.models.reranker_model_path} ...")
                self.reranker = BGEReranker(model_path=settings.models.reranker_model_path)
                logger.info("BGE 重排模型加载完成")
            except Exception as e:
                logger.warning(f"BGE 重排模型加载失败: {e}")
                logger.warning("将仅使用传统重排方法")

        # 6. 时间衰减权重
        self.recency_weights = settings.retriever.recency_weights

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """
        执行混合检索

        实现 BaseRetriever 的抽象方法

        Args:
            query: 查询文本
            top_k: 返回结果数量
            **kwargs: 其他参数（如阈值调整等）

        Returns:
            (context_str, results, metadata)
            - context_str: 拼接的上下文文本
            - results: 详细的结果列表，每个结果包含：
              {
                  "entity": Dict,              # Milvus 实体
                  "distance": float,            # 距离
                  "similarity": float,          # 相似度
                  "title": str,                 # 标题
                  "department": str,            # 部门
                  "time": str,                  # 时间
                  "content": str,               # 内容
                  "composite_score": float,     # 综合评分（重排后）
                  "rerank_features": Dict       # 重排特征（可选）
              }
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

            # 放宽检索数量，为后续筛选和重排准备
            search_limit = top_k * 3
            raw_results = self.milvus_client.search(
                vectors=query_vec.tolist(),
                top_k=search_limit,
                output_fields=["text", "department", "metadata"],
            )

            if not raw_results or not raw_results[0]:
                result = (
                    "未在知识库中找到相关信息。",
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
                # Milvus 2.x 中，distance 对应余弦相似度
                entity = hit.get("entity", hit)
                metadata = entity.get("metadata", {})

                processed_hit = {
                    "entity": entity,
                    "distance": 1 - hit.get("distance", 0),
                    "similarity": hit.get("distance", 0),
                    # 直接提取常用字段，方便上层使用
                    "title": metadata.get("title", "无标题"),
                    "department": entity.get("department", "未知部门"),
                    "time": metadata.get("time", "未知时间"),
                    "content": entity.get("text", ""),
                }
                processed_results.append(processed_hit)

            # 4. 混合阈值筛选
            filtered_results = self._hybrid_threshold_filter(processed_results, query)

            # 5. 混合重排
            reranked_results = self._hybrid_rerank(query, filtered_results)

            # 6. 截取最终结果
            final_results = reranked_results[:min(top_k, len(reranked_results))]

            # 7. 构建上下文
            context_str = self.build_context(query, final_results)

            # 8. 准备元数据
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
                # 添加详细来源信息
                "sources": [
                    {
                        "rank": i + 1,
                        "title": r.get("title", "无标题"),
                        "department": r.get("department", "未知部门"),
                        "time": r.get("time", "未知时间"),
                        "similarity": r.get("similarity", 0),
                        "composite_score": r.get("composite_score", 0),
                    }
                    for i, r in enumerate(final_results)
                ],
            }

            # 9. 更新缓存
            if self.cache_enabled:
                self._update_cache(cache_key, context_str, final_results, metadata)

            return context_str, final_results, metadata

        except Exception as e:
            import traceback
            logger.error(f"[HybridRetriever] 检索失败: {e}")
            logger.error(traceback.format_exc())
            # 返回包含必要字段的 metadata
            return f"检索服务暂时不可用: {str(e)}", [], {
                "query": query,
                "error": str(e),
                "num_results": 0,
                "avg_similarity": 0.0,
                "threshold_applied": self.min_similarity,
                "cache_hit": False,
                "sources": []
            }

    # ==================== 内部方法 ====================

    def _hybrid_threshold_filter(self, results: List[Dict], query: str) -> List[Dict]:
        """
        混合阈值筛选策略

        策略步骤：
        1. 使用基础阈值筛选
        2. 如果结果太少，动态降低阈值
        3. 确保至少有最小结果数量
        """
        if not results:
            return []

        # 步骤1：基础阈值筛选
        threshold = self.min_similarity
        filtered = [r for r in results if r["similarity"] >= threshold]

        # 步骤2：分析结果分布
        similarities = [r["similarity"] for r in results[:10]]  # 只看前10个
        mean_sim = np.mean(similarities) if similarities else 0

        # 步骤3：动态调整
        if len(filtered) < self.min_results:
            if mean_sim < threshold:
                # 如果整体相似度较低，适当降低阈值
                adaptive_threshold = max(threshold * 0.8, mean_sim - 0.1)
                adaptive_threshold = max(0.3, adaptive_threshold)  # 保底阈值

                logger.info(f"阈值动态调整: {threshold:.3f} → {adaptive_threshold:.3f}")
                filtered = [r for r in results if r["similarity"] >= adaptive_threshold]

        # 返回筛选后的结果
        return filtered if len(filtered) > 0 else results

    def _hybrid_rerank(self, query: str, results: List[Dict]) -> List[Dict]:
        """
        混合重排策略

        基于多个特征综合评分（移除部门权威性，所有部门信息平等对待）：
        1. 向量相似度 (60%)
        2. 时效性 (30%)
        3. 内容长度 (10%)
        4. BGE 重排模型 (如果可用)
        """
        if len(results) <= 1:
            return results

        # 如果存在BGE重排模型，优先使用它进行重排
        if self.reranker is not None:
            logger.info(f"使用 BGE 重排模型对 {len(results)} 个结果进行重排...")
            reranked_results = self.reranker.rerank(query, results)

            # 将BGE重排得分与其他特征结合起来
            for hit in reranked_results:
                # 使用BGE重排得分作为主要的composite_score
                hit["composite_score"] = hit.get("rerank_score", 0.0)

                # 保存原有相似度特征用于显示
                hit["original_similarity"] = hit.get("similarity", 0.0)

                # 计算其他特征用于更精细的排序
                current_time = datetime.now()
                time_str = hit["entity"].get("metadata", {}).get("time", "")
                recency = self._calculate_recency(time_str, current_time)

                text_len = len(hit["entity"].get("text", ""))
                length_score = min(1.0, text_len / 1500)

                # 保存特征值供后续使用
                hit["rerank_features"] = {
                    "similarity": hit.get("original_similarity", 0.0),
                    "recency": recency,
                    "length": length_score,
                    "bge_rerank": hit.get("rerank_score", 0.0)
                }

            return reranked_results
        else:
            # 如果没有BGE重排模型，则使用传统的混合重排策略
            current_time = datetime.now()
            features = {key: [] for key in self.rerank_weights.keys()}

            for hit in results:
                # 1. 相似度特征
                features["similarity"].append(hit["similarity"])

                # 2. 时效性特征
                time_str = hit["entity"].get("metadata", {}).get("time", "")
                recency = self._calculate_recency(time_str, current_time)
                features["recency"].append(recency)

                # 3. 内容长度特征
                text_len = len(hit["entity"].get("text", ""))
                length_score = min(1.0, text_len / 1500)
                features["length"].append(length_score)

            # 归一化特征
            norm_features = {}
            for key, values in features.items():
                norm_features[key] = self._normalize_features(values)

            # 计算综合评分
            for i, hit in enumerate(results):
                composite_score = 0
                for key, weight in self.rerank_weights.items():
                    composite_score += norm_features[key][i] * weight

                hit["composite_score"] = composite_score
                hit["rerank_features"] = {
                    key: norm_features[key][i] for key in self.rerank_weights.keys()
                }

            # 按综合评分降序排序
            results.sort(key=lambda x: x["composite_score"], reverse=True)
            return results

    def _calculate_recency(self, time_str: str, current_time: datetime) -> float:
        """计算时效性分数"""
        if not time_str:
            return 0.5

        try:
            # 尝试解析常见时间格式
            formats = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d"]
            item_time = None

            for fmt in formats:
                try:
                    item_time = datetime.strptime(time_str.split()[0], fmt)
                    break
                except:
                    continue

            if item_time:
                # 计算时间衰减（越近分数越高）
                days_diff = (current_time - item_time).days
                if days_diff < 0:  # 未来时间
                    return 0.5
                elif days_diff <= 7:  # 一周内
                    return 1.0
                elif days_diff <= 30:  # 一月内
                    return 0.9
                elif days_diff <= 90:  # 三月内
                    return 0.7
                elif days_diff <= 365:  # 一年内
                    return 0.5
                else:  # 超过一年
                    return 0.3
        except:
            pass

        return 0.5  # 默认值

    def _normalize_features(self, values: List[float]) -> List[float]:
        """归一化特征值到0-1范围"""
        if not values:
            return values

        min_val, max_val = min(values), max(values)

        if max_val - min_val < 1e-6:  # 避免除以0
            return [0.5] * len(values)

        return [(v - min_val) / (max_val - min_val) for v in values]

    # ==================== 静态工厂方法 ====================

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "HybridVectorRetriever":
        """
        从配置创建实例

        Args:
            config: 配置字典

        Returns:
            HybridVectorRetriever 实例
        """
        return cls(config=config)

    @classmethod
    def from_settings(cls) -> "HybridVectorRetriever":
        """
        从项目配置创建实例

        Returns:
            HybridVectorRetriever 实例
        """
        return cls()