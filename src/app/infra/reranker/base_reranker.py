"""
BGE Reranker 基础客户端 - Infra 层

提供线程安全的单例模式，只负责模型推理，
不引入任何 LangChain 概念（不认识 Document）
"""

import threading
from pathlib import Path
from typing import List

from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from src.config.setting import settings
from src.app.infra.utils.logger import get_logger

logger = get_logger(__name__)


class BaseRerankerClient:
    """
    BaseRerankerClient 客户端 - 线程安全单例

    只暴露 computer_score 接口，不认识 LangChain Document
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """初始化"""
        if self._initialized:
            return

        self.model_path = Path(settings.models.reranker_model_path)

        logger.info(f"加载 BGE Reranker 模型: {self.model_path} ...")

        # 使用 LangChain HuggingFaceCrossEncoder
        self.reranker = HuggingFaceCrossEncoder(
            model_name=str(self.model_path),
        )

        self._initialized = True
        logger.info("BGE Reranker 模型加载完成")

    def compute_score(self, query: str, texts: List[str]) -> List[float]:
        """
        计算查询与每个文本的相关性得分

        Args:
            query: 查询文本
            texts: 待评分的文本列表

        Returns:
            每个文本的相关性得分列表
        """
        if not texts:
            return []

        # 创建查询-文档对
        pairs = [[query, text] for text in texts]

        # 执行评分
        scores = self.reranker.score(pairs)

        return [float(score) for score in scores]

    @classmethod
    def get_instance(cls) -> "BaseRerankerClient":
        """
        获取单例实例

        Returns:
            BaseReranker 实例
        """
        return cls()


def get_reranker_client() -> BaseRerankerClient:
    """
    获取 Reranker 客户端单例的便捷函数

    Returns:
        BaseReranker 实例
    """
    return BaseRerankerClient.get_instance()