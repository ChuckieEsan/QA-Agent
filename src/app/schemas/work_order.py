"""工单领域数据模型"""

from pydantic import BaseModel, Field


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