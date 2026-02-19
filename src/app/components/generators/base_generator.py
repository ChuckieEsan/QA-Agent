"""
生成器抽象基类
提供统一的文本生成接口
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, AsyncGenerator, Any


class BaseGenerator(ABC):
    """
    生成器抽象基类

    所有生成器实现都应该继承此类
    """

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        history: Optional[List[Dict]] = None,
        **kwargs
    ) -> str:
        """
        同步生成文本

        Args:
            prompt: 用户提示词
            system_message: 系统消息（可选）
            history: 对话历史（可选）
            **kwargs: 其他生成参数

        Returns:
            生成的文本
        """
        pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        history: Optional[List[Dict]] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        流式生成文本

        Args:
            prompt: 用户提示词
            system_message: 系统消息（可选）
            history: 对话历史（可选）
            **kwargs: 其他生成参数

        Yields:
            生成的文本片段
        """
        pass

    @abstractmethod
    async def generate_with_validation(
        self,
        prompt: str,
        validation_criteria: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        生成并验证文本

        Args:
            prompt: 用户提示词
            validation_criteria: 验证标准
            **kwargs: 其他生成参数

        Returns:
            {
                "text": str,                    # 生成的文本
                "quality_score": float,         # 质量分数 (0-1)
                "passed_validation": bool,      # 是否通过验证
                "validation_details": Dict      # 验证详情
            }
        """
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """
        初始化生成器资源
        """
        pass