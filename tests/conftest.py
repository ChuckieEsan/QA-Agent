"""
tests/conftest.py
测试配置和共享 fixture
"""

import pytest
import os
import sys
from pathlib import Path
from datetime import timedelta
from unittest.mock import MagicMock, Mock

# 添加项目根目录到 Python 路径
sys.path.append(os.getcwd())


# ================= 环境隔离 =================
@pytest.fixture(autouse=True)
def mock_environment(monkeypatch):
    """
    自动隔离环境变量，确保测试不依赖真实环境
    """
    # 屏蔽所有敏感环境变量
    monkeypatch.setenv("DASHSCOPE_API_KEY", "fake-test-key-123456")

    # 设置测试专用路径
    monkeypatch.setenv("TEST_MODE", "true")

    # 移除可能存在的真实环境变量
    for key in list(os.environ.keys()):
        if key.endswith("_API_KEY") and key not in ["DASHSCOPE_API_KEY"]:
            monkeypatch.delenv(key, raising=False)


# ================= 配置模拟 =================
@pytest.fixture
def mock_settings(monkeypatch):
    """
    模拟应用配置
    """
    from app.core.config import settings

    # 使用 monkeypatch 修改 settings 的属性
    # 注意：这里假设 settings 使用 @property 或可以被直接修改

    # 修改路径配置
    monkeypatch.setattr(settings.paths, "milvus_db_path", ":memory:")  # 使用内存数据库
    monkeypatch.setattr(settings.paths, "data_dir", Path("/tmp/test_data"))

    # 修改模型配置
    monkeypatch.setattr(
        settings.models, "embedding_model_path", Path("/tmp/test_models/bge-m3")
    )

    # 修改检索器配置
    monkeypatch.setattr(settings.retriever, "base_threshold", 0.65)
    monkeypatch.setattr(settings.retriever, "min_results", 2)
    monkeypatch.setattr(settings.retriever, "max_results", 10)

    # 修改向量数据库配置
    monkeypatch.setattr(settings.vectordb, "collection_name", "test_collection")

    return settings


# ================= 模型模拟 =================
@pytest.fixture
def mock_embedding_model(mocker):
    """
    模拟 SentenceTransformer 模型
    返回固定维度的假向量
    """
    mock_model = MagicMock()

    # 模拟 encode 方法
    def mock_encode(texts, **kwargs):
        batch_size = len(texts) if isinstance(texts, list) else 1
        # 返回 BGE-M3 的 1024 维向量
        return [[0.1 * i for i in range(1024)]] * batch_size

    mock_model.encode = Mock(side_effect=mock_encode)

    # Patch 实际导入位置
    mocker.patch("app.services.retriever.SentenceTransformer", return_value=mock_model)

    return mock_model


# ================= 数据库模拟 =================
@pytest.fixture
def mock_milvus_client(mocker):
    """
    模拟 Milvus 客户端
    """
    mock_client = MagicMock()

    # 模拟 search 方法返回结构化数据
    def mock_search(collection_name, data, limit, output_fields, **kwargs):
        # 返回符合 HybridVectorRetriever 预期的数据结构
        return [
            [
                {
                    "distance": 0.1,
                    "entity": {
                        "text": "标题：测试案例1\n部门：测试局\n时间：2024-01-01\n市民诉求：测试问题1\n官方回复：测试回答1\n来源链接：http://example.com",
                        "department": "测试局",
                        "metadata": {
                            "title": "测试案例1",
                            "question": "测试问题1",
                            "answer": "测试回答1",
                            "time": "2024-01-01",
                            "url": "http://example.com",
                        },
                    },
                },
                {
                    "distance": 0.2,
                    "entity": {
                        "text": "标题：测试案例2\n部门：另一个部门\n时间：2024-01-02\n市民诉求：测试问题2\n官方回复：测试回答2\n来源链接：http://example.com/2",
                        "department": "另一个部门",
                        "metadata": {
                            "title": "测试案例2",
                            "question": "测试问题2",
                            "answer": "测试回答2",
                            "time": "2024-01-02",
                            "url": "http://example.com/2",
                        },
                    },
                },
            ]
        ]

    mock_client.search = Mock(side_effect=mock_search)

    # 模拟其他必要方法
    mock_client.has_collection = Mock(return_value=True)
    mock_client.list_collections = Mock(return_value=["test_collection"])

    # Patch MilvusClient
    mocker.patch("app.services.retriever.MilvusClient", return_value=mock_client)

    return mock_client


# ================= 检索器实例 =================
@pytest.fixture
def hybrid_retriever_instance(mocker, mock_embedding_model, mock_milvus_client):
    """
    返回一个完全模拟的 HybridVectorRetriever 实例
    跳过真实初始化过程
    """
    # 重置单例
    from app.services.retriever import HybridVectorRetriever

    HybridVectorRetriever._instance = None
    HybridVectorRetriever._is_initialized = False

    # 创建实例，但跳过 _init_resources
    with mocker.patch.object(HybridVectorRetriever, "_init_resources"):
        retriever = HybridVectorRetriever()
        retriever._is_initialized = True  # 标记为已初始化

    # 手动设置模拟的依赖
    retriever.embed_model = mock_embedding_model
    retriever.client = mock_milvus_client
    retriever.collection = "test_collection"

    # 设置配置参数
    retriever.base_threshold = 0.65
    retriever.min_results = 2
    retriever.max_results = 10
    retriever.rerank_weights = {
        "similarity": 0.4,
        "recency": 0.3,
        "authority": 0.2,
        "length": 0.1,
    }
    retriever.dept_authority = {"省政府": 1.0, "市政府": 0.8, "default": 0.5}

    # 初始化缓存
    retriever.cache = {}
    retriever.cache_ttl = timedelta(minutes=5)

    return retriever


# ================= 测试数据 =================
@pytest.fixture
def sample_search_results():
    """
    提供样本搜索结果数据
    """
    return [
        [
            {
                "distance": 0.1,
                "entity": {
                    "text": "案例1的完整上下文",
                    "department": "市政府",
                    "metadata": {
                        "title": "标题1",
                        "question": "问题1",
                        "answer": "回答1",
                        "time": "2024-01-10",
                        "url": "http://example.com/1",
                    },
                },
            },
            {
                "distance": 0.2,
                "entity": {
                    "text": "案例2的完整上下文",
                    "department": "人社局",
                    "metadata": {
                        "title": "标题2",
                        "question": "问题2",
                        "answer": "回答2",
                        "time": "2023-12-01",
                        "url": "http://example.com/2",
                    },
                },
            },
        ]
    ]


# ================= 测试标记配置 =================
def pytest_configure(config):
    """配置 pytest 标记"""
    config.addinivalue_line("markers", "integration: 标记为集成测试（需要外部服务）")
    config.addinivalue_line("markers", "slow: 标记为慢速测试")
    config.addinivalue_line("markers", "unit: 标记为单元测试（默认）")


def pytest_addoption(parser):
    """添加命令行选项"""
    parser.addoption(
        "--run-integration", action="store_true", default=False, help="运行集成测试"
    )


def pytest_collection_modifyitems(config, items):
    """根据命令行选项过滤测试"""
    if not config.getoption("--run-integration"):
        skip_integration = pytest.mark.skip(
            reason="需要 --run-integration 选项来运行集成测试"
        )
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)
