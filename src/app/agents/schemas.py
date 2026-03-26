"""
Agent 数据模型定义
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentResponse(BaseModel):
    """Agent 响应模型"""

    messages: List[Any] = Field(default=[], description="消息列表")
    final_reply: str = Field(default="", description="最终回复")
    classification: Dict[str, Any] = Field(default={}, description="分类结果")
    work_order_id: Optional[str] = Field(default=None, description="工单ID")
    confidence_score: float = Field(default=0.0, description="置信度评分")