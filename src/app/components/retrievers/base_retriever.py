"""
检索器基类 - 定义统一的检索接口

设计目标：
1. 提供统一的检索接口，支持多种检索策略
2. 支持缓存机制
3. 支持检索质量评估
4. 便于扩展新的检索器实现
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta


class BaseRetriever(ABC):
    """
    检索器抽象基类

    所有检索器实现都应该继承此类，并实现抽象方法
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化检索器

        Args:
            config: 检索器配置字典，包含：
                - top_k: 默认返回结果数量
                - cache_enabled: 是否启用缓存
                - cache_ttl: 缓存过期时间（秒）
                - min_similarity: 最小相似度阈值
        """
        self.config = config or {}

        # 缓存配置
        self.cache_enabled: bool = self.config.get("cache_enabled", True)
        self.cache_ttl: int = self.config.get("cache_ttl", 300)  # 默认5分钟
        self.cache: Dict[str, Dict[str, Any]] = {}

        # 检索参数
        self.default_top_k: int = self.config.get("top_k", 5)
        self.min_similarity: float = self.config.get("min_similarity", 0.5)

        # 初始化状态
        self._is_initialized: bool = False

    @abstractmethod
    def initialize(self) -> None:
        """
        初始化检索器资源

        子类必须实现此方法，用于：
        - 加载模型
        - 连接数据库
        - 初始化其他资源
        """
        pass

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """
        执行检索操作

        Args:
            query: 查询文本
            top_k: 返回结果数量，None时使用默认值
            **kwargs: 其他检索参数

        Returns:
            Tuple 包含：
                - context_str: RAG上下文字符串
                - results: 检索结果列表
                - metadata: 元数据（包含检索时间、数量等信息）

        Example:
            >>> context, results, metadata = retriever.retrieve("查询文本", top_k=5)
            >>> print(f"检索到 {len(results)} 个结果")
        """
        pass

    @abstractmethod
    def retrieve_with_details(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        执行详细检索，返回结构化的完整信息

        Args:
            query: 查询文本
            top_k: 返回结果数量
            **kwargs: 其他检索参数

        Returns:
            Dict 包含：
                - query: 原始查询
                - context: RAG上下文
                - sources: 来源列表（包含相似度、部门等信息）
                - metadata: 元数据
                - confidence: 置信度分数
        """
        pass

    # ==================== 缓存管理方法 ====================

    def _get_cache_key(self, query: str, top_k: int, **kwargs) -> str:
        """
        生成缓存键

        Args:
            query: 查询文本
            top_k: 返回结果数量
            **kwargs: 其他参数

        Returns:
            缓存键字符串
        """
        # 基础键：query + top_k
        key_parts = [query, str(top_k)]

        # 添加其他参数（如果有）
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")

        return "_".join(key_parts)

    def _check_cache(
        self,
        cache_key: str
    ) -> Optional[Tuple[str, List[Dict[str, Any]], Dict[str, Any]]]:
        """
        检查缓存

        Args:
            cache_key: 缓存键

        Returns:
            缓存的结果（如果有且未过期），否则返回 None
        """
        if not self.cache_enabled:
            return None

        if cache_key not in self.cache:
            return None

        cache_entry = self.cache[cache_key]

        # 检查是否过期
        now = datetime.now()
        cache_time = cache_entry["timestamp"]
        ttl = timedelta(seconds=self.cache_ttl)

        if now - cache_time > ttl:
            # 缓存过期，删除
            del self.cache[cache_key]
            return None

        # 返回缓存结果
        return (
            cache_entry["context"],
            cache_entry["results"],
            cache_entry["metadata"]
        )

    def _update_cache(
        self,
        cache_key: str,
        context: str,
        results: List[Dict[str, Any]],
        metadata: Dict[str, Any]
    ) -> None:
        """
        更新缓存

        Args:
            cache_key: 缓存键
            context: RAG上下文
            results: 检索结果
            metadata: 元数据
        """
        if not self.cache_enabled:
            return

        self.cache[cache_key] = {
            "context": context,
            "results": results,
            "metadata": metadata,
            "timestamp": datetime.now()
        }

        # 限制缓存大小（防止内存泄漏）
        max_cache_size = self.config.get("max_cache_size", 100)
        if len(self.cache) > max_cache_size:
            # 删除最早的缓存项
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

    def clear_cache(self) -> None:
        """清空缓存"""
        self.cache.clear()

    # ==================== 辅助方法 ====================

    def calculate_confidence(self, results: List[Dict[str, Any]]) -> float:
        """
        计算检索置信度

        Args:
            results: 检索结果列表

        Returns:
            置信度分数（0-1）
        """
        if not results:
            return 0.0

        # 默认实现：基于相似度和数量
        similarities = [
            r.get("similarity", r.get("score", 0))
            for r in results
        ]

        avg_similarity = sum(similarities) / len(similarities)

        # 数量因子：结果越多，置信度越高（但边际递减）
        num_factor = 1 - 0.5 ** len(results)

        return min(1.0, avg_similarity * num_factor)

    def build_context(
        self,
        query: str,
        results: List[Dict[str, Any]],
        context_template: Optional[str] = None
    ) -> str:
        """
        构建RAG上下文字符串

        Args:
            query: 原始查询
            results: 检索结果列表
            context_template: 可选的上下文模板

        Returns:
            构建好的上下文字符串
        """
        if not results:
            return "未找到相关案例。"

        # 默认模板
        if context_template is None:
            context_parts = [f"用户查询：{query}", f"检索到 {len(results)} 个相关案例：\n"]

            for i, hit in enumerate(results):
                # 获取相似度
                similarity = hit.get("similarity", hit.get("score", 0))
                confidence = (
                    "高" if similarity > 0.7 else ("中" if similarity > 0.5 else "低")
                )

                # 获取文本内容
                text = hit.get("entity", {}).get("text", hit.get("text", ""))

                context_parts.append(f"\n--- 案例 {i+1} (相似度: {similarity:.1%}, 置信度: {confidence}) ---")
                context_parts.append(text)

            context_parts.append("\n--- 回答指导 ---")
            context_parts.append("请基于以上案例信息，准确、专业地回应用户查询。")
            context_parts.append("如果案例与查询不完全匹配，请说明差异并提供最相关的信息。")
            context_parts.append("引用具体案例时，注明来源部门和时间。")

            return "\n".join(context_parts)

        # 使用自定义模板
        return context_template.format(
            query=query,
            results=results,
            num_results=len(results)
        )

    # ==================== 工厂方法 ====================

    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseRetriever":
        """
        从配置创建检索器实例

        Args:
            config: 配置字典

        Returns:
            检索器实例
        """
        pass

    # ==================== 上下文管理器支持 ====================

    def __enter__(self):
        """支持上下文管理器"""
        if not self._is_initialized:
            self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文时清理资源"""
        self.clear_cache()
        return False
