"""
Embedding 模型单例

提供 Embedding 模型的统一加载和访问
"""

import threading
from pathlib import Path
from typing import List, Union
import numpy as np

from sentence_transformers import SentenceTransformer

from src.config.setting import settings
from src.app.infra.utils import get_device
from src.app.infra.utils.logger import get_logger

logger = get_logger(__name__)


class BaseEmbedding:
    """
    Embedding 模型单例类

    确保整个应用只加载一次 Embedding 模型，节省内存
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
        """初始化 Embedding 模型"""
        if self._initialized:
            return

        # 获取模型路径和设备
        self.model_path = Path(settings.models.embedding_model_path)
        self.device = get_device()
        self.dimension = settings.models.embedding_size

        # 加载模型
        logger.info(f"加载 Embedding 模型: {self.model_path} ...")
        self.model = SentenceTransformer(
            str(self.model_path),
            device=self.device
        )
        logger.info(f"Embedding 模型加载完成，设备: {self.device}")

        self._initialized = True

    def encode(
        self,
        texts: Union[str, List[str]],
        normalize_embeddings: bool = True,
        **kwargs
    ) -> np.ndarray:
        """
        将文本编码为向量

        Args:
            texts: 单个文本或文本列表
            normalize_embeddings: 是否归一化向量
            **kwargs: 其他参数

        Returns:
            编码后的向量数组
        """
        if isinstance(texts, str):
            texts = [texts]

        return self.model.encode(
            texts,
            normalize_embeddings=normalize_embeddings,
            **kwargs
        )

    @property
    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return self._initialized

    @classmethod
    def get_instance(cls) -> "BaseEmbedding":
        """
        获取单例实例

        Returns:
            BaseEmbedding 实例
        """
        return cls()
    
