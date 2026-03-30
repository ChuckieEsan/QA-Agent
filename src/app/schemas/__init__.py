"""
统一数据模型定义
按领域聚合：classification, validator, work_order, agent
"""

from .agent import AgentResponse
from .classification import GovRequestClassifiedResult, GovRequestType
from .validator import GovAnswerValidatedResult, LLMValidationResult
from .work_order import WorkOrderData, WorkOrderResult

__all__ = [
    # Agent
    "AgentResponse",
    # Classification
    "GovRequestType",
    "GovRequestClassifiedResult",
    # Validator
    "LLMValidationResult",
    "GovAnswerValidatedResult",
    # Work Order
    "WorkOrderData",
    "WorkOrderResult",
]