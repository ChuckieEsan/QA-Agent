"""
记忆组件抽象基类
提供统一的对话记忆管理接口
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any


class BaseMemory(ABC):
    """
    记忆组件抽象基类

    所有记忆实现都应该继承此类
    """

    @abstractmethod
    def add_message(self, message: Dict[str, any]) -> None:
        """
        添加消息到记忆

        Args:
            message: 消息字典，格式：{"role": "user|assistant", "content": str}
        """
        pass

    @abstractmethod
    def get_messages(self, limit: Optional[int] = None) -> List[Dict[str, any]]:
        """
        获取记忆中的消息

        Args:
            limit: 返回的消息数量限制（None 表示全部）

        Returns:
            消息列表
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        清空记忆
        """
        pass

    @abstractmethod
    def get_context(self, max_tokens: Optional[int] = None) -> str:
        """
        获取对话上下文（用于 Prompt 构建）

        Args:
            max_tokens: 最大 token 数限制

        Returns:
            格式化的对话上下文字符串
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        保存记忆到文件

        Args:
            path: 保存路径
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        从文件加载记忆

        Args:
            path: 加载路径
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, any]:
        """
        获取记忆统计信息

        Returns:
            {
                "total_messages": int,
                "user_messages": int,
                "assistant_messages": int,
                "total_tokens": int,
                "created_at": str,
                "last_updated": str
            }
        """
        pass
