"""
Agent 抽象基类
定义统一的 Agent 接口
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime


class BaseAgent(ABC):
    """
    Agent 抽象基类

    所有 Agent 实现都应该继承此类
    """

    def __init__(self, name: str = "Agent"):
        """
        初始化 Agent

        Args:
            name: Agent 名称
        """
        self.name = name
        self.created_at = datetime.now()
        self._initialized = False

    @abstractmethod
    async def process(
        self,
        query: str,
        context: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        处理查询的核心方法

        Args:
            query: 用户查询
            context: 上下文信息（可选）
            **kwargs: 其他参数

        Returns:
            处理结果字典
        """
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """
        初始化 Agent 资源

        子类应该实现此方法，用于：
        - 初始化依赖组件
        - 预热模型
        - 加载配置
        """
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        获取 Agent 状态

        Returns:
            状态信息字典
        """
        pass

    def is_initialized(self) -> bool:
        """
        检查 Agent 是否已初始化

        Returns:
            是否已初始化
        """
        return self._initialized
