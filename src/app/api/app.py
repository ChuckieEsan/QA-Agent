"""
GovPulse 应用工厂
创建 FastAPI 应用实例
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.app.api.routes import router
from src.app.infra.utils.logger import get_logger

logger = get_logger(__name__)


def create_app() -> FastAPI:
    """
    创建 FastAPI 应用实例

    Returns:
        FastAPI 应用
    """
    # 创建 FastAPI 应用
    app = FastAPI(
        title="GovPulse API",
        description="泸州市政务智能问答系统 API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # 配置 CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册路由
    app.include_router(router)

    # 配置事件处理器
    @app.on_event("startup")
    async def startup_event():
        """应用启动事件"""
        logger.info("🚀 GovPulse API 服务启动")
        logger.info(f"📄 API 文档: http://localhost:8000/docs")
        logger.info(f"📝 ReDoc 文档: http://localhost:8000/redoc")

    @app.on_event("shutdown")
    async def shutdown_event():
        """应用关闭事件"""
        logger.info("🛑 GovPulse API 服务关闭")

    @app.exception_handler(404)
    async def not_found_handler(request, exc):
        """404 错误处理"""
        return {"error": "Not Found", "path": str(request.url)}

    return app
