"""
Reranker 基础设施层

提供线程安全的单例 Reranker 客户端
"""

from .base_reranker import BaseRerankerClient, get_reranker_client

__all__ = ["BaseRerankerClient", "get_reranker_client"]