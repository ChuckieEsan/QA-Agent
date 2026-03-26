"""MCP 工具定义 - 使用 LangChain @tool 装饰器"""

import uuid
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from src.app.infra.utils.logger import get_logger

logger = get_logger(__name__)


class WorkOrderData(BaseModel):
    """工单数据模型"""

    user_id: str = Field(default="", description="提交人ID")
    user_phone: str = Field(default="", description="联系方式")
    title: str = Field(..., description="标题")
    content: str = Field(..., description="详细描述")
    department: str = Field(default="", description="责任部门")
    elements: str = Field(default="", description="五大核心要素")


class WorkOrderResult(BaseModel):
    """工单创建结果模型"""

    success: bool = Field(default=True, description="是否成功")
    order_id: str = Field(..., description="工单ID")
    message: str = Field(default="", description="返回消息")
    status: str = Field(default="UNASSIGNED", description="工单状态")


@tool
def create_work_order(order_data: WorkOrderData) -> WorkOrderResult:
    """
    创建政务工单

    当置信度低于阈值时，调用此工具创建工单交给人工处理。

    Args:
        order_data: 工单数据

    Returns:
        WorkOrderResult: 工单创建结果
    """
    logger.info(f"[Tool] create_work_order called with: {order_data.title}")

    # TODO: 实际调用 MCP Server 或后端服务
    # 这里暂时返回模拟数据

    order_id = f"WO{uuid.uuid4().hex[:12].upper()}"

    return WorkOrderResult(
        success=True,
        order_id=order_id,
        message="工单创建成功",
        status="UNASSIGNED",
    )


def get_mcp_tools():
    """
    获取可用的工具列表

    Returns:
        工具列表
    """
    return [
        create_work_order,
    ]