"""
GovPulse API 启动文件
项目根目录的启动入口
"""

import uvicorn
from src.app.api.app import create_app

# 创建应用实例
app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=6666,
        reload=True,
        log_level="info"
    )
