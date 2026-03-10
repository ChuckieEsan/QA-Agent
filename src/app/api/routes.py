"""
GovPulse API 路由定义
包含所有 API 路由，不包含服务器启动逻辑
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio

from src.app.agents import ainvoke
from src.app.agents.state import ProcessStatus
from src.app.infra.utils.logger import get_logger
from src.app.infra.db.milvus_db import MilvusDBClient

logger = get_logger(__name__)

# 创建 API 路由器
router = APIRouter(prefix="/api", tags=["govpulse"])


# ==================== 数据模型 ====================


class ChatRequest(BaseModel):
    """聊天请求模型"""

    query: str = Field(..., description="用户查询", min_length=1, max_length=1000)
    history: List[Dict[str, str]] = Field(
        default=[],
        description="对话历史，格式: [{'role': 'user', 'content': '...'}, ...]",
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
    classification: Dict[str, Any] = Field(..., description="分类结果")
    sources: List[SourceItem] = Field(..., description="检索来源")
    quality_score: float = Field(..., description="质量评分", ge=0.0, le=1.0)
    retrieval_time: float = Field(..., description="检索耗时（秒）")
    steps: int = Field(default=1, description="执行步数")
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
        "history": [],
        "top_k": 5
    }
    ```

    **返回说明**:
    - `answer`: 生成的回答
    - `classification`: 问政类型分类（建议/投诉/求助/咨询）
    - `sources`: 检索到的案例来源
    - `quality_score`: 回答质量评分（0-1）
    - `retrieval_time`: 检索耗时（秒）
    """
    try:
        logger.info(f"💬 收到聊天请求: {request.query[:30]}...")

        # 调用 LangGraph MultiAgent 工作流
        state = await ainvoke(request.query)

        # 从新状态结构中提取数据
        classification = state.get("classification", {})
        status = state.get("status", ProcessStatus.PENDING)

        # 根据状态确定回答
        if status == ProcessStatus.WORK_ORDER_CREATED:
            answer = "您的诉求已收到，由于需要人工核实处理，我们已为您创建工单，稍后会有工作人员与您联系。"
        else:
            answer = state.get("generated_response", "")

        # 转换检索结果
        sources = []
        for i, source in enumerate(state.get("retrieved_knowledge", [])):
            sources.append(SourceItem(
                rank=i + 1,
                similarity=source.get("similarity", 0.0),
                department=source.get("department", "未知部门"),
                title=source.get("title", "无标题"),
                time=source.get("time", "未知时间"),
                composite_score=source.get("composite_score", 0.0),
            ))

        # 构建响应
        response = ChatResponse(
            answer=answer,
            classification={
                "type": classification.get("request_type", "未知"),
                "urgency_level": classification.get("request_urgency", "一般"),
                "status": status.value if isinstance(status, ProcessStatus) else str(status),
            },
            sources=sources,
            quality_score=state.get("confidence_score", 0.0),
            retrieval_time=0.0,
            steps=1,
            timestamp=datetime.now().isoformat(),
        )

        logger.info(f"✅ 聊天响应完成，检索到 {len(response.sources)} 个来源")
        return response

    except Exception as e:
        logger.error(f"❌ 聊天请求处理失败: {e}")
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
