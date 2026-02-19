"""
分类器抽象基类
提供统一的文本分类接口
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from enum import Enum


class GovRequestType(Enum):
    """
    问政请求类型枚举

    泸州市网络问政平台的四种主要类型：
    1. 建议：对政府工作提出改进建议、意见
    2. 投诉：反映政府部门或工作人员的问题、不当行为
    3. 求助：请求政府帮助解决个人或家庭困难
    4. 咨询：询问政策、流程、办事指南等信息
    """
    ADVICE = "advice"       # 建议
    COMPLAINT = "complaint" # 投诉
    HELP = "help"           # 求助
    CONSULT = "consult"     # 咨询


class BaseClassifier(ABC):
    """
    分类器抽象基类

    所有分类器实现都应该继承此类
    """

    @abstractmethod
    async def classify_gov_request(
        self,
        text: str,
        **kwargs
    ) -> Dict[str, any]:
        """
        分类问政请求

        Args:
            text: 市民诉求文本
            **kwargs: 其他分类参数

        Returns:
            {
                "type": "advice|complaint|help|consult",  # 问政类型
                "confidence": float,                       # 置信度 (0-1)
                "keywords": List[str],                     # 关键词
                "reasoning": str                           # 分类理由
            }
        """
        pass

    @abstractmethod
    async def classify_batch(
        self,
        texts: List[str],
        **kwargs
    ) -> List[Dict[str, any]]:
        """
        批量分类问政请求

        Args:
            texts: 市民诉求文本列表
            **kwargs: 其他分类参数

        Returns:
            分类结果列表
        """
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """
        初始化分类器资源
        """
        pass
