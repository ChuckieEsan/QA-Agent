import threading
import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from src.config.setting import settings
from src.app.infra.utils import get_device


class HybridVectorRetriever:
    """
    纯混合策略向量检索器
    结合固定阈值+动态调整+重排的完整流程
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(HybridVectorRetriever, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_is_initialized", False):
            return

        print("🔄 [HybridRetriever] 初始化混合策略检索器...")
        self._init_resources()
        self._is_initialized = True
        print("✅ [HybridRetriever] 初始化完成")

    def _init_resources(self):
        """初始化核心资源"""
        # 1. 加载Embedding模型
        self.device = get_device()
        print(f"📥 加载Embedding模型: {settings.models.embedding_model_path} ...")
        self.embed_model = SentenceTransformer(
            str(settings.models.embedding_model_path), device=self.device
        )

        # 2. 连接向量数据库
        print(f"🔌 连接Milvus: {settings.vectordb.db_path} ...")
        self.client = MilvusClient(str(settings.vectordb.db_path))
        self.collection = settings.vectordb.collection_name

        # 3. 混合策略配置
        self.base_threshold = settings.retriever.base_threshold
        self.min_results = settings.retriever.min_results
        self.max_results = settings.retriever.max_results

        # 4. 重排权重配置
        self.rerank_weights = {
            "similarity": settings.retriever.weight_similarity,
            "recency": settings.retriever.weight_recency,
            "authority": settings.retriever.weight_authority,
            "length": settings.retriever.weight_length,
        }

        # TODO: 5. 部门权威性映射
        self.dept_authority = settings.retriever.department_authority

        # TODO: 6. 缓存. 后续可以改成 Redis
        self.cache = {}
        self.cache_ttl = timedelta(minutes=5)

    def retrieve(self, query: str, top_k: int = None) -> Tuple[str, List[Dict], Dict]:
        """
        混合策略检索主函数

        参数:
            query: 查询文本
            top_k: 返回结果数量（None时使用配置默认值）

        返回:
            (context_str, results, metadata)
        """
        start_time = datetime.now()

        if top_k is None:
            top_k = min(self.max_results, max(self.min_results, 5))

        # 1. 检查缓存
        cache_key = f"{query}_{top_k}"
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if datetime.now() - cache_entry["timestamp"] < self.cache_ttl:
                print(f"🔄 使用缓存结果: {cache_key[:30]}...")
                return (
                    cache_entry["context"],
                    cache_entry["results"],
                    cache_entry["metadata"],
                )

        try:
            # 2. 向量检索
            query_vec = self.embed_model.encode([query], normalize_embeddings=True)

            # 放宽检索数量，为后续筛选和重排准备
            search_limit = top_k * 3
            raw_results = self.client.search(
                collection_name=self.collection,
                data=query_vec,
                limit=search_limit,
                output_fields=["text", "department", "metadata"],
            )

            if not raw_results or not raw_results[0]:
                result = ("未在知识库中找到相关信息。", [], {})
                self._update_cache(cache_key, result, start_time)
                return result

            # 3. 转换结果并计算相似度
            processed_results = []
            for hit in raw_results[0]:
                # 特别注意, 在 Milvus 2.x 版本中, distance 对应的就是余弦相似度
                processed_hit = {
                    "entity": hit["entity"],
                    "distance": 1 - hit["distance"],
                    "similarity": hit["distance"],
                }
                processed_results.append(processed_hit)

            # 4. 混合阈值筛选
            filtered_results = self._hybrid_threshold_filter(processed_results, query)

            # 5. 混合重排
            reranked_results = self._hybrid_rerank(query, filtered_results)

            # 6. 截取最终结果
            final_results = reranked_results[: min(top_k, len(reranked_results))]

            # 7. 构建上下文
            context_str = self._build_context(query, final_results)

            # 8. 准备元数据
            metadata = {
                "query": query,
                "retrieval_time": (datetime.now() - start_time).total_seconds(),
                "num_results": len(final_results),
                "num_raw_results": len(processed_results),
                "avg_similarity": (
                    np.mean([r["similarity"] for r in final_results])
                    if final_results
                    else 0
                ),
                "threshold_applied": self.base_threshold,
                "cache_hit": False,
            }

            # 9. 更新缓存
            result = (context_str, final_results, metadata)
            self._update_cache(cache_key, result, start_time)

            return result

        except Exception as e:
            print(f"⚠️ [HybridRetriever] 检索失败: {e}")
            return (f"检索服务暂时不可用: {str(e)}", [], {})

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
        base_threshold = self.base_threshold
        filtered = [r for r in results if r["similarity"] >= base_threshold]

        # 步骤2：分析结果分布
        similarities = [r["similarity"] for r in results[:10]]  # 只看前10个
        mean_sim = np.mean(similarities) if similarities else 0

        # 步骤3：动态调整
        if len(filtered) < self.min_results:
            if mean_sim < base_threshold:
                # 如果整体相似度较低，适当降低阈值
                adaptive_threshold = max(base_threshold * 0.8, mean_sim - 0.1)
                adaptive_threshold = max(0.3, adaptive_threshold)  # 保底阈值

                print(
                    f"📊 阈值动态调整: {base_threshold:.3f} → {adaptive_threshold:.3f}"
                )
                filtered = [r for r in results if r["similarity"] >= adaptive_threshold]

        # 步骤4：保底机制
        if len(filtered) < self.min_results:
            # 返回相似度最高的前几个结果，但标记为低置信度
            sorted_results = sorted(
                results, key=lambda x: x["similarity"], reverse=True
            )
            filtered = sorted_results[: self.min_results]
            for r in filtered:
                r["low_confidence"] = True

        # TODO: 暂时不考虑过滤
        # return filtered
        return results

    def _hybrid_rerank(self, query: str, results: List[Dict]) -> List[Dict]:
        """
        混合重排策略

        基于多个特征综合评分：
        1. 向量相似度
        2. 时效性
        3. 部门权威性
        4. 内容质量
        """
        if len(results) <= 1:
            return results

        current_time = datetime.now()
        features = {key: [] for key in self.rerank_weights.keys()}

        for hit in results:
            # 1. 相似度特征
            features["similarity"].append(hit["similarity"])

            # 2. 时效性特征
            time_str = hit["entity"].get("metadata", {}).get("time", "")
            recency = self._calculate_recency(time_str, current_time)
            features["recency"].append(recency)

            # 3. 部门权威性特征
            dept = hit["entity"].get("department", "")
            authority = self.dept_authority.get(dept, self.dept_authority["default"])
            features["authority"].append(authority)

            # TODO: 4. 内容长度特征. 需要进一步优化
            text_len = len(hit["entity"].get("text", ""))
            length_score = min(1.0, text_len / 1500)  # 1500字为理想长度
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

        # TODO: 按综合评分重排 (暂时不考虑重排)
        # return sorted(results, key=lambda x: x["composite_score"], reverse=True)
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

    def _build_context(self, query: str, results: List[Dict]) -> str:
        """构建RAG上下文"""
        if not results:
            return "未找到相关案例。"

        context_parts = [f"用户查询：{query}", f"检索到 {len(results)} 个相关案例：\n"]

        for i, hit in enumerate(results):
            similarity = hit["similarity"]
            confidence = (
                "高" if similarity > 0.7 else ("中" if similarity > 0.5 else "低")
            )

            # 如果有重排评分，显示综合评分
            if "composite_score" in hit:
                composite_score = hit["composite_score"]
                score_info = f"(相似度: {similarity:.1%}, 综合评分: {composite_score:.3f}, 置信度: {confidence})"
            else:
                score_info = f"(相似度: {similarity:.1%}, 置信度: {confidence})"

            # 直接使用已构建的RAG上下文
            rag_text = hit["entity"].get("text", "")

            context_parts.append(f"\n--- 案例 {i+1} {score_info} ---")
            context_parts.append(rag_text)

        # 添加回答指导
        context_parts.append("\n--- 回答指导 ---")
        context_parts.append("请基于以上案例信息，准确、专业地回应用户查询。")
        context_parts.append("如果案例与查询不完全匹配，请说明差异并提供最相关的信息。")
        context_parts.append("引用具体案例时，请注明来源部门和时间。")

        return "\n".join(context_parts)

    def _update_cache(self, cache_key: str, result: Tuple, timestamp: datetime):
        """更新缓存"""
        context_str, results, metadata = result

        # 只缓存成功的查询
        if results:
            self.cache[cache_key] = {
                "context": context_str,
                "results": results,
                "metadata": {**metadata, "cache_hit": True},
                "timestamp": timestamp,
            }

            # 限制缓存大小
            if len(self.cache) > 100:
                # 删除最旧的缓存项
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]

    def search_with_details(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        详细搜索接口（包含元数据）

        参数:
            query: 查询文本
            top_k: 返回结果数量

        返回:
            包含完整信息的字典
        """
        context_str, results, metadata = self.retrieve(query, top_k)

        # 提取关键信息
        sources = []
        for i, hit in enumerate(results):
            entity = hit["entity"]
            meta = entity.get("metadata", {})

            sources.append(
                {
                    "rank": i + 1,
                    "similarity": hit["similarity"],
                    "department": entity.get("department", "未知部门"),
                    "title": meta.get("title", "无标题"),
                    "time": meta.get("time", "未知时间"),
                    "composite_score": hit.get("composite_score", 0),
                }
            )

        return {
            "query": query,
            "context": context_str,
            "sources": sources,
            "metadata": metadata,
            "num_sources": len(sources),
            "confidence": self._calculate_confidence(results),
        }

    def _calculate_confidence(self, results: List[Dict]) -> float:
        """计算检索置信度"""
        if not results:
            return 0.0

        # 基于相似度和数量计算置信度
        similarities = [r["similarity"] for r in results]
        avg_similarity = np.mean(similarities)

        # 数量因子：结果越多，置信度越高（但边际递减）
        num_factor = 1 - 0.5 ** len(results)

        # 综合置信度
        confidence = avg_similarity * num_factor
        return min(1.0, confidence)

    # TODO: 缓存管理接口
    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        print("🧹 缓存已清空")


# 工具函数
def retrieve_with_details(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    获取RAG上下文及详细信息
    """
    retriever = HybridVectorRetriever()
    return retriever.search_with_details(query, top_k)


def get_retriever_instance() -> HybridVectorRetriever:
    """
    获取检索器单例实例
    """
    return HybridVectorRetriever()
