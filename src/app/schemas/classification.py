"""分类领域数据模型"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class GovRequestType(Enum):
    """
    问政请求类型枚举

    泸州市网络问政平台的四种主要类型：
    1. 建议：对政府工作提出改进建议、意见
    2. 投诉：反映政府部门或工作人员的问题、不当行为
    3. 求助：请求政府帮助解决个人或家庭困难
    4. 咨询：询问政策、流程、办事指南等信息
    5. 其他：与问政内容无关
    """

    ADVICE = "advice"  # 建议
    COMPLAINT = "complaint"  # 投诉
    HELP = "help"  # 求助
    CONSULT = "consult"  # 咨询
    OTHER = "other"  # 其他

    @property
    def chinese(self) -> str:
        """根据成员名返回中文描述"""
        return {
            'ADVICE': '建议',
            'COMPLAINT': '投诉',
            'HELP': '求助',
            'CONSULT': '咨询',
            'OTHER': '其他',
        }[self.name]


class GovRequestClassifiedResult(BaseModel):
    """问政请求分类结果模型"""

    request_type: GovRequestType = Field(description="问政请求类型")
    request_department: Optional[str] = Field(
        default="待确定",
        description="问政请求精确对应的市、区、县级主管部门或者相关单位。如果不确定具体的相关单位，请直接输出'待确定'",
    )
    request_city_department: Optional[str] = Field(
        default="待检索",
        description="问政请求对应的市级主管部门。如果不确定具体的管辖部门，请直接输出 '待检索'，不要瞎猜。",
    )

    class Config:
        populate_by_name = True