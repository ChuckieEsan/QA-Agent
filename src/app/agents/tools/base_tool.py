"""
ReAct 工具层
将现有组件封装为 ReAct 工具，支持动态调用
"""

import inspect
from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BaseTool(ABC):
    """
    工具抽象基类

    所有 ReAct 工具都应该继承此类，实现统一的执行接口
    """

    name: str              # 工具名称
    description: str       # 工具描述

    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        执行工具

        Args:
            **kwargs: 工具执行参数

        Returns:
            Dict[str, Any]: 执行结果字典

        Raises:
            Exception: 工具执行异常
        """
        pass

    def get_schema(self) -> Dict[str, Any]:
        """
        获取工具 Schema (供 LLM 调用)

        Returns:
            工具 Schema 信息，包括：
            - name: 工具名称
            - description: 工具描述
            - parameters: 参数说明（可选）
        """
        schema = {
            "name": self.name,
            "description": self.description
        }
        return schema

    def get_required_parameters(self) -> List[str]:
        """
        通过反射获取工具的必填参数

        Returns:
            必填参数名称列表（即没有默认值的参数）
        """
        required_params = []

        try:
            sig = inspect.signature(self.execute)

            for name, param in sig.parameters.items():
                if name in ('self', 'kwargs'):
                    continue

                if param.default == inspect.Parameter.empty:
                    required_params.append(name)

        except Exception:
            pass

        return required_params

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"

