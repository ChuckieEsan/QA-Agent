"""
质量验证器基类
定义统一的质量验证接口
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseValidator(ABC):
    """
    质量验证器抽象基类

    所有验证器实现都应该继承此类
    """

    @abstractmethod
    async def validate(
        self,
        answer: str,
        query: str,
        context: str
    ) -> Dict[str, Any]:
        """
        验证回答质量

        Args:
            answer: 生成的回答
            query: 用户查询
            context: 检索上下文

        Returns:
            质量校验结果字典，包含：
            - relevance_score: 相关性分数 (0-1)
            - accuracy_score: 准确性分数 (0-1)
            - attribution_score: 来源标注分数 (0-1)
            - compliance_score: 合规性分数 (0-1)
            - overall_score: 综合分数 (0-1)
            - suggestion: 优化建议（可选）
        """
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """
        初始化验证器资源
        """
        pass
