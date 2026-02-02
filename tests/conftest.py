import pytest
import os
from unittest.mock import MagicMock

# 1. 自动隔离环境变量
# autouse=True: 每个测试运行前都会自动执行，不需要手动引用
@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    """
    强制覆盖环境变量，防止测试读取你本地真实的 .env 文件。
    """
    monkeypatch.setenv("QWEN_API_KEY", "sk-fake-test-key")
    monkeypatch.setenv("MILVUS_DB_PATH", "./data/test_db/test.db")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-fake-test-key")

# 2. 模拟 Settings
@pytest.fixture
def mock_settings(mock_env):
    """
    返回一个加载了假环境变量的配置对象
    """
    from app.core.config import settings
    return settings

# 3. 模拟 Embedding 模型 (避免加载 2GB 的文件)
@pytest.fixture
def mock_sentence_transformer(mocker):
    """
    Mock 掉 SentenceTransformer，让它返回固定的假向量。
    """
    mock_model = MagicMock()
    # 模拟 encode 方法，返回一个全 0 的 1024 维向量
    mock_model.encode.return_value = [[0.1] * 1024] 
    
    # 这里的路径 'app.services.rag_engine.SentenceTransformer' 需要根据你实际引用的位置写
    # 如果你在 services/rag_engine.py 里 import 了它，就要 patch 那里的引用
    mocker.patch("sentence_transformers.SentenceTransformer", return_value=mock_model)
    return mock_model

# 4. 模拟 Milvus 客户端
@pytest.fixture
def mock_milvus_client(mocker):
    mock_client = MagicMock()
    # 模拟 search 返回结果
    mock_client.search.return_value = [[
        {
            "id": 1, 
            "distance": 0.8, 
            "entity": {"text": "测试文档内容", "department": "测试局"}
        }
    ]]
    mocker.patch("pymilvus.MilvusClient", return_value=mock_client)
    return mock_client