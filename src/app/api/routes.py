"""
GovPulse API 路由定义
包含所有 API 路由，不包含服务器启动逻辑
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

from src.app.agents import ainvoke
from src.app.infra.utils.logger import get_logger
from src.app.infra.db.milvus_db import MilvusDBClient

logger = get_logger(__name__)

# 创建 API 路由器
router = APIRouter(prefix="/api", tags=["govpulse"])


# ==================== 数据模型 ====================


class ChatRequest(BaseModel):
    """聊天请求模型"""

    query: str = Field(..., description="用户查询", min_length=1, max_length=1000)
    session_id: str = Field(
        default="default",
        description="会话ID，用于多轮对话状态管理",
    )
    history: List[Dict[str, str]] = Field(
        default=[],
        description="对话历史（已废弃，由 LangGraph checkpointer 自动管理）",
        deprecated=True,
    )
    top_k: int = Field(default=5, ge=1, le=20, description="检索结果数量")


class SourceItem(BaseModel):
    """检索来源项"""

    rank: int = Field(..., description="排名")
    similarity: float = Field(..., description="相似度")
    department: str = Field(..., description="部门名称")
    title: str = Field(..., description="标题")
    time: str = Field(..., description="时间")
    composite_score: float = Field(default=0.0, description="综合评分")


class ChatResponse(BaseModel):
    """聊天响应模型"""

    answer: str = Field(..., description="生成的回答")
    classification: Dict[str, Any] = Field(default={}, description="分类结果")
    sources: List[SourceItem] = Field(default=[], description="检索来源")
    quality_score: float = Field(default=0.0, description="质量评分", ge=0.0, le=1.0)
    retrieval_time: float = Field(default=0.0, description="检索耗时（秒）")
    steps: int = Field(default=1, description="执行步数")
    work_order_id: Optional[str] = Field(default=None, description="工单ID（如果有）")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class HealthResponse(BaseModel):
    """健康检查响应"""

    status: str = "ok"
    version: str = "1.0.0"
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# ==================== API 路由 ====================


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口"""
    return HealthResponse(status="ok")


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    政务问答接口

    该接口接收用户查询，返回智能生成的回答和检索来源。

    **示例请求**:
    ```json
    {
        "query": "雨露计划什么时候发放？",
        "session_id": "user-123"
    }
    ```

    **返回说明**:
    - `answer`: 生成的回答
    - `classification`: 问政类型分类（建议/投诉/求助/咨询）
    - `sources`: 检索到的案例来源（暂未实现）
    - `quality_score`: 回答质量评分（0-1）
    - `retrieval_time`: 检索耗时（秒）
    - `work_order_id`: 工单ID（如果触发兜底）
    """
    try:
        logger.info(f"💬 收到聊天请求: {request.query[:30]}..., session: {request.session_id}")

        # 调用 LangGraph MultiAgent 工作流
        state = await ainvoke(request.query, session_id=request.session_id)

        # 从新状态结构中提取数据
        classification = state.get("classification", {}) or {}
        work_order_id = state.get("work_order_id")

        # 获取最终回复
        answer = state.get("final_reply", "抱歉，服务暂时不可用。")

        # 如果有工单，添加提示信息
        if work_order_id:
            logger.info(f"✅ 工单已创建: {work_order_id}")

        # 构建响应
        response = ChatResponse(
            answer=answer,
            classification={
                "type": classification.get("request_type", "未知") if isinstance(classification, dict) else "未知",
                "request_department": classification.get("request_department", "") if isinstance(classification, dict) else "",
            },
            sources=[],  # TODO: 实现来源返回
            quality_score=state.get("confidence_score", 0.0),
            retrieval_time=0.0,
            steps=1,
            work_order_id=work_order_id,
            timestamp=datetime.now().isoformat(),
        )

        logger.info(f"✅ 聊天响应完成")
        return response

    except Exception as e:
        logger.error(f"❌ 聊天请求处理失败: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_stats():
    """获取系统统计信息"""
    try:
        client = MilvusDBClient()
        stats = client.get_collection_stats()  # 不需要参数

        return {
            "total_documents": stats.get("row_count", 0),
            "collection_name": "gov_cases",
            "status": "active",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))