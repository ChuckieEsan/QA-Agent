"""工具注册表 - 管理所有可用工具"""

from typing import Dict, Any, Optional, Callable
from langchain_core.tools import BaseTool
from src.app.agents.tools.mcp_tools import get_available_tools
from src.app.infra.utils.logger import get_logger

logger = get_logger(__name__)


class ToolRegistry:
    """
    工具注册表

    管理所有可用的 MCP 工具
    """

    _instance = None
    _tools: Dict[str, BaseTool] = {}

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """初始化工具注册表"""
        logger.info("[ToolRegistry] 初始化工具注册表...")

        # 加载所有可用工具
        tools = get_available_tools()
        for tool_func in tools:
            self._tools[tool_func.name] = tool_func

        logger.info(f"[ToolRegistry] 已加载 {len(self._tools)} 个工具")

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """
        获取指定名称的工具

        Args:
            name: 工具名称

        Returns:
            工具实例，如果不存在则返回 None
        """
        return self._tools.get(name)

    def call_tool(self, name: str, args: Dict[str, Any]) -> Any:
        """
        调用指定工具

        Args:
            name: 工具名称
            args: 工具参数

        Returns:
            工具执行结果
        """
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found")

        logger.info(f"[ToolRegistry] Calling tool: {name}")
        result = tool.invoke(args)
        return result

    def list_tools(self) -> list:
        """
        列出所有可用工具

        Returns:
            工具名称列表
        """
        return list(self._tools.keys())