"""MCP 工具定义 - 使用 LangChain @tool 装饰器"""

from typing import Dict, Any, Optional
from langchain_core.tools import tool
from src.app.infra.utils.logger import get_logger

logger = get_logger(__name__)


@tool
def create_work_order(order_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    创建政务工单

    当置信度低于阈值时，调用此工具创建工单交给人工处理。

    Args:
        order_data: 工单数据，包含以下字段:
            - user_id: 提交人ID
            - user_phone: 联系方式
            - title: 标题
            - content: 详细描述
            - department: 责任部门
            - elements: 五大核心要素

    Returns:
        包含 order_id 的字典
    """
    logger.info(f"[Tool] create_work_order called with: {order_data.get('title', 'N/A')}")

    # TODO: 实际调用 MCP Server 或后端服务
    # 这里暂时返回模拟数据
    import uuid
    order_id = f"WO{uuid.uuid4().hex[:12].upper()}"

    return {
        "success": True,
        "order_id": order_id,
        "message": "工单创建成功",
        "status": "UNASSIGNED",
    }


@tool
def query_knowledge_base(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    检索知识库

    根据用户诉求检索相关的政策、办事指南等信息。

    Args:
        query: 用户查询
        top_k: 返回结果数量

    Returns:
        检索结果列表
    """
    logger.info(f"[Tool] query_knowledge_base called: {query[:30]}...")

    # TODO: 实际调用知识库检索服务
    # 这里暂时返回模拟数据
    return {
        "query": query,
        "results": [],
        "count": 0,
    }


@tool
def query_department_responsibility(department: str) -> Dict[str, Any]:
    """
    查询部门权责清单

    查询某个部门的职责范围和办理事项。

    Args:
        department: 部门名称

    Returns:
        部门权责信息
    """
    logger.info(f"[Tool] query_department_responsibility called: {department}")

    # TODO: 实际调用权责清单服务
    return {
        "department": department,
        "responsibilities": [],
        "services": [],
    }


def get_available_tools():
    """
    获取可用的工具列表

    Returns:
        工具列表
    """
    return [
        create_work_order,
        query_knowledge_base,
        query_department_responsibility,
    ]