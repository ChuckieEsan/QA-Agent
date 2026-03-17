"""
工具注册表
不再维护单例类，直接聚合返回 List[BaseTool] 给 Agent 使用
"""

from typing import List
from langchain_core.tools import BaseTool
from src.app.infra.utils.logger import get_logger

# 导入本地工具
from src.app.agents.tools.local_tools import (
    classify_gov_request_tool,
    retrieve_powers_tool,
    retrieve_cases_tool,
    validate_answer_tool
)

# 导入MCP 工具
from src.app.agents.tools.mcp_tools import get_mcp_tools

logger = get_logger(__name__)

def get_all_tools() -> List[BaseTool]:
    """
    动态获取当前 Agent 可用的所有工具集合。
    每次创建 Agent 或处理请求时调用此函数，确保 MCP 工具列表是最新的。
    """
    # 1. 加载本地固化的核心业务工具
    tools: List[BaseTool] =[
        classify_gov_request_tool,
        retrieve_powers_tool,
        retrieve_cases_tool,
        validate_answer_tool
    ]
    
    # 2. 动态追加 MCP 远程工具 (比如网关层的 create_work_order 等)
    try:
        mcp_tools = get_mcp_tools()
        if mcp_tools:
            tools.extend(mcp_tools)
            logger.info(f"[ToolRegistry] 成功挂载 {len(mcp_tools)} 个 MCP 远程工具")
    except Exception as e:
        logger.error(f"[ToolRegistry] 动态加载 MCP 工具失败，已降级为纯本地工具模式。错误: {e}")
        
    logger.info(f"[ToolRegistry] 工具集装配完成，共提供 {len(tools)} 个工具供 Agent 调用。")
    return tools