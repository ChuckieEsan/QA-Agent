"""MCP 工具定义 - 使用 LangChain @tool 装饰器"""

import uuid
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

    order_id = f"WO{uuid.uuid4().hex[:12].upper()}"

    return {
        "success": True,
        "order_id": order_id,
        "message": "工单创建成功",
        "status": "UNASSIGNED",
    }


def get_available_tools():
    """
    获取可用的工具列表

    Returns:
        工具列表
    """
    return [
        create_work_order,
    ]