"""
ReAct 工具模块
提供工具类和工具注册表

使用装饰器模式注册工具：
    from src.app.agents.tools.registry import ToolRegistry

    @ToolRegistry.register()
    class MyTool(BaseTool):
        name = "my_tool"
        description = "我的工具"

        async def execute(self, **kwargs) -> dict:
            return {"result": "xxx"}

使用 ReactAgent：
    from src.app.agents import ReactAgent
    from src.app.agents.tools import ToolRegistry

    tools = {
        "retrieve": ToolRegistry.get_instance("retrieve"),
        "generate": ToolRegistry.get_instance("generate"),
        "classify": ToolRegistry.get_instance("classify"),
        "validate": ToolRegistry.get_instance("validate"),
    }
    agent = ReactAgent(tools, max_steps=5)
    result = await agent.process("2024年泸州雨露计划补贴标准")
"""

from src.app.agents.tools.base_tool import BaseTool
from src.app.agents.tools.retrieval_tool import RetrievalTool
from src.app.agents.tools.generation_tool import GenerationTool
from src.app.agents.tools.classification_tool import ClassificationTool
from src.app.agents.tools.validation_tool import ValidationTool
from src.app.agents.tools.registry import ToolRegistry

__all__ = [
    "BaseTool",
    "RetrievalTool",
    "GenerationTool",
    "ClassificationTool",
    "ValidationTool",
    "ToolRegistry",
]
