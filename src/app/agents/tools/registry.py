"""
工具注册表模式
便于扩展、管理和发现 ReAct 工具

使用装饰器注册工具后，可直接获取工具实例：
    from src.app.agents.tools.registry import ToolRegistry

    @ToolRegistry.register()
    class MyTool(BaseTool):
        name = "my_tool"
        description = "我的工具"

        async def execute(self, **kwargs) -> dict:
            return {"result": "xxx"}

    # 直接获取工具实例
    tool = ToolRegistry.get_instance("my_tool")
"""

from typing import Dict, Type, Optional, Any
from src.app.agents.tools.base_tool import BaseTool


class ToolRegistry:
    """
    工具注册表 - 便于扩展和管理 ReAct 工具

    提供工具的注册、查找和实例化功能，支持:
    - 通过装饰器自动注册工具（同时创建实例）
    - 通过名称获取工具实例
    - 支持自定义工具扩展

    使用示例：
        from src.app.agents.tools.registry import ToolRegistry

        @ToolRegistry.register()
        class MyTool(BaseTool):
            name = "my_tool"
            description = "我的工具"

            async def execute(self, **kwargs) -> dict:
                return {"result": "xxx"}

        # 获取工具实例
        tool = ToolRegistry.get_instance("my_tool")
    """

    _registry: Dict[str, BaseTool] = {}

    @classmethod
    def register(cls, name: Optional[str] = None):
        """
        注册工具类装饰器（同时创建实例）

        Args:
            name: 注册名称（可选，不指定则使用工具类的 name 属性）

        Returns:
            装饰器函数

        Example:
            @ToolRegistry.register()
            class MyTool(BaseTool):
                name = "my_tool"
                description = "我的工具"
        """
        def decorator(tool_cls: Type[BaseTool]) -> Type[BaseTool]:
            tool_name = name or getattr(tool_cls, "name", tool_cls.__name__.lower().replace("tool", ""))
            if tool_name in cls._registry:
                raise ValueError(f"工具 '{tool_name}' 已经注册")
            # 直接创建实例
            instance = tool_cls()
            cls._registry[tool_name] = instance
            return tool_cls
        return decorator

    @classmethod
    def get_instance(cls, name: str) -> BaseTool:
        """
        获取工具实例

        Args:
            name: 工具名称

        Returns:
            工具实例

        Raises:
            KeyError: 工具未注册
        """
        if name not in cls._registry:
            raise KeyError(f"工具 '{name}' 未注册。可用工具: {list(cls._registry.keys())}")
        return cls._registry[name]

    @classmethod
    def get_class(cls, name: str) -> Type[BaseTool]:
        """
        获取工具类（用于类型检查）

        Args:
            name: 工具名称

        Returns:
            工具类

        Raises:
            KeyError: 工具未注册
        """
        instance = cls.get_instance(name)
        return type(instance)

    @classmethod
    def list_all(cls) -> Dict[str, BaseTool]:
        """
        列出所有已注册的工具实例

        Returns:
            工具名称到工具实例的映射
        """
        return dict(cls._registry)

    @classmethod
    def list_classes(cls) -> Dict[str, Type[BaseTool]]:
        """
        列出所有已注册的工具类

        Returns:
            工具名称到工具类的映射
        """
        return {name: type(instance) for name, instance in cls._registry.items()}
