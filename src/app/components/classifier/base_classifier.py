"""
分类器抽象基类
提供统一的文本分类接口
"""

from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from enum import Enum


class GovRequestUrgency(Enum):
    """
    问政请求紧急情况枚举
    """
    MAJOR = "major" # 紧急
    NORMAL = "normal" # 一般
    MINOR = "minor" # 轻微 


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
    ADVICE = "advice"       # 建议
    COMPLAINT = "complaint" # 投诉
    HELP = "help"           # 求助
    CONSULT = "consult"     # 咨询
    OTHER = "other"         # 其他

    
class GovRequestClassifiedResult(BaseModel):
    request_type: GovRequestType = Field(description="问政请求类型")
    request_urgency: GovRequestUrgency = Field(description="紧急程度")
    
    class Config:
        # 允许字段别名（例如 'requestType' -> 'request_type'）
        populate_by_name = True


class BaseClassifier(ABC):
    """
    分类器抽象基类

    所有分类器实现都应该继承此类
    """

    @abstractmethod
    def classify(
        self,
        text: str,
    ) -> GovRequestClassifiedResult:
        """
        分类问政请求

        Args:
            text: 市民诉求文本

        Returns: 
            GovRequestClassifiedResult
        """
        pass
    
    