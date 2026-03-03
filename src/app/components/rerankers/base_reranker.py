"""
重排器基类 - 定义统一的重排接口

设计目标：
1. 提供统一的重排接口，支持多种重排模型
2. 便于扩展新的重排器实现
3. 统一输入输出格式
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from pathlib import Path


class BaseReranker(ABC):
    """
    重排器抽象基类

    所有重排器实现都应该继承此类，并实现抽象方法
    """

    @abstractmethod
    def __init__(self, model_path: Path = None):
        """
        初始化重排器

        Args:
            model_path: 模型路径，如果为 None 则使用默认配置
        """
        pass

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        对文档进行重排

        Args:
            query: 查询文本
            documents: 待重排的文档列表
            top_k: 返回前 K 个结果，如果为 None 则返回全部结果

        Returns:
            重排后的文档列表，按相关性降序排列，每个文档增加了 'rerank_score' 字段
        """
        pass

    @abstractmethod
    def compute_score(self, query: str, text: str) -> float:
        """
        计算单个查询与文本的相关性得分

        Args:
            query: 查询文本
            text: 待评分文本

        Returns:
            相关性得分
        """
        pass