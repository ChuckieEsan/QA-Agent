"""
测试 API 路由
"""

import pytest
from fastapi.testclient import TestClient
from src.app.api.app import create_app


@pytest.fixture
def client():
    """创建测试客户端"""
    app = create_app()
    return TestClient(app)


class TestAPIRoutes:
    """测试 API 路由"""

    def test_health_check(self, client):
        """测试健康检查接口"""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_stats_endpoint(self, client):
        """测试统计接口"""
        response = client.get("/api/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_documents" in data
        assert "collection_name" in data

    @pytest.mark.integration
    @pytest.mark.skip(reason="需要真实 LLM 服务")
    def test_chat_endpoint(self, client):
        """测试聊天接口（集成测试）"""
        response = client.post(
            "/api/chat",
            json={
                "query": "雨露计划什么时候发放？",
                "history": [],
                "top_k": 5
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
