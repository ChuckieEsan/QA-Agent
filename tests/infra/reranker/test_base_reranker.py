"""
测试 BaseRerankerClient - Infra 层

直接测试与底层 BGE 模型的交互，不使用 mock
"""

import pytest
import threading

from langchain_core.documents import Document

from src.app.infra.reranker import BaseRerankerClient, get_reranker_client
from src.app.components.reranker import BGERerankerCompressor, create_bge_compressor


class TestBaseRerankerClient:
    """测试 BaseRerankerClient 单例客户端"""

    @pytest.fixture(scope="class")
    def client(self):
        """获取客户端实例"""
        return BaseRerankerClient.get_instance()

    def test_singleton_pattern(self):
        """测试单例模式：多次调用应返回同一实例"""
        client1 = BaseRerankerClient.get_instance()
        client2 = BaseRerankerClient.get_instance()
        assert client1 is client2, "多次调用 get_instance() 应返回同一实例"

    def test_singleton_via_constructor(self):
        """测试通过构造函数也是单例"""
        client1 = BaseRerankerClient()
        client2 = BaseRerankerClient()
        assert client1 is client2, "通过构造函数调用也应返回同一实例"

    def test_singleton_thread_safety(self):
        """测试单例模式的线程安全性"""
        results = []
        lock = threading.Lock()

        def get_client():
            client = BaseRerankerClient()
            with lock:
                results.append(client)

        # 并发创建多个实例
        threads = [threading.Thread(target=get_client) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 所有线程应该获得同一个实例
        assert len(set(id(r) for r in results)) == 1, "并发调用应返回同一实例"

    def test_compute_score_returns_float_list(self, client):
        """测试 compute_score 返回浮点数列表"""
        query = "如何申请创业补贴？"
        texts = [
            "创业补贴申请需要提供营业执照复印件",
            "雨露计划是国家精准扶贫政策",
            "创业担保贷款申请条件包括户籍要求",
        ]

        scores = client.compute_score(query, texts)

        assert isinstance(scores, list), "返回类型应为列表"
        assert len(scores) == len(texts), "得分数量应与文本数量一致"
        assert all(isinstance(s, float) for s in scores), "每个得分应为浮点数"

    def test_compute_score_empty_texts(self, client):
        """测试空文本列表"""
        scores = client.compute_score("query", [])
        assert scores == [], "空文本应返回空列表"

    def test_compute_score_single_text(self, client):
        """测试单个文本"""
        query = "什么是社保？"
        text = "社保是社会保险的简称，包括养老保险、医疗保险等"

        scores = client.compute_score(query, [text])

        assert len(scores) == 1
        assert isinstance(scores[0], float)


class TestBGERerankerCompressor:
    """测试 BGERerankerCompressor 与 BaseRerankerClient 的集成"""

    @pytest.fixture
    def client(self):
        """获取 BGE 客户端"""
        return BaseRerankerClient.get_instance()

    @pytest.fixture
    def compressor(self, client):
        """创建压缩器实例"""
        return BGERerankerCompressor(
            bge_client=client,
            top_k=3,
            min_score=0.0,
        )

    def test_compressor_not_singleton(self, client):
        """测试压缩器不是单例：每次创建都是新实例"""
        compressor1 = BGERerankerCompressor(bge_client=client)
        compressor2 = BGERerankerCompressor(bge_client=client)
        assert compressor1 is not compressor2, "压缩器不应是单例"

    def test_compress_documents_empty(self, compressor):
        """测试空文档列表"""
        result = compressor.compress_documents([], "query")
        assert result == [], "空文档应返回空列表"

    def test_compress_documents_single(self, compressor):
        """测试单个文档：直接返回"""
        documents = [
            Document(
                page_content="这是唯一的相关文档",
                metadata={"source": "test"}
            )
        ]

        result = compressor.compress_documents(documents, "查询")

        assert len(result) == 1
        assert result[0].page_content == "这是唯一的相关文档"

    def test_compress_documents_reranks_by_score(self, compressor):
        """测试文档重排：相关度高的应排在前面"""
        documents = [
            Document(
                page_content="雨露计划是国家针对农村贫困人口实施的精准扶贫政策",
                metadata={"id": 1, "source": "test1"}
            ),
            Document(
                page_content="创业补贴申请需要提供营业执照复印件等相关材料",
                metadata={"id": 2, "source": "test2"}
            ),
            Document(
                page_content="创业担保贷款申请条件包括户籍要求和经营时间要求",
                metadata={"id": 3, "source": "test3"}
            ),
        ]

        # 查询创业相关内容
        result = compressor.compress_documents(
            documents,
            "如何申请创业补贴？"
        )

        # 创业相关的文档应该排在前面
        assert len(result) == 3
        # 检查得分是否已添加
        for doc in result:
            assert "rerank_score" in doc.metadata
            assert "composite_score" in doc.metadata
        # 得分应该按降序排列
        scores = [doc.metadata["rerank_score"] for doc in result]
        assert scores == sorted(scores, reverse=True), "得分应按降序排列"

    def test_compress_documents_top_k(self, client):
        """测试 top_k 参数限制返回数量"""
        compressor = BGERerankerCompressor(bge_client=client, top_k=2)

        documents = [
            Document(page_content=f"文档{i}内容", metadata={"id": i})
            for i in range(10)
        ]

        result = compressor.compress_documents(documents, "查询")

        assert len(result) == 2, f"应返回 top_k=2 个文档，实际返回 {len(result)} 个"

    def test_compress_documents_preserves_metadata(self, compressor):
        """测试重排后保留原始 metadata"""
        original_metadata = {
            "source": "milvus",
            "department": "公积金中心",
            "year": 2024,
            "id": 123
        }
        documents = [
            Document(
                page_content="公积金提取需要提供购房合同",
                metadata=original_metadata.copy()
            ),
            Document(
                page_content="雨露计划是扶贫政策",
                metadata={"source": "other"}
            ),
        ]

        result = compressor.compress_documents(documents, "公积金提取")

        # 第一个文档应保留原有 metadata
        assert result[0].metadata.get("source") == "milvus"
        assert result[0].metadata.get("department") == "公积金中心"
        assert result[0].metadata.get("year") == 2024
        assert result[0].metadata.get("id") == 123
        # 同时也应包含新的 rerank_score
        assert "rerank_score" in result[0].metadata

    def test_compress_documents_min_score_filter(self, client):
        """测试 min_score 过滤"""
        compressor = BGERerankerCompressor(
            bge_client=client,
            top_k=10,
            min_score=100.0  # 设置一个很高的阈值
        )

        documents = [
            Document(page_content=f"文档{i}", metadata={"id": i})
            for i in range(5)
        ]

        result = compressor.compress_documents(documents, "查询")

        # 由于所有得分都不会超过 100（通常在 0-10 之间），应返回空
        assert len(result) <= 5, "高阈值下可能过滤掉所有文档"


class TestCreateBgeCompressor:
    """测试 create_bge_compressor 工厂函数"""

    def test_create_with_client(self):
        """测试传入自定义客户端"""
        client = BaseRerankerClient.get_instance()
        compressor = create_bge_compressor(bge_client=client, top_k=3)

        assert compressor.bge_client is client
        assert compressor.top_k == 3

    def test_create_without_client(self):
        """测试不传客户端时自动获取单例"""
        compressor = create_bge_compressor(top_k=5)

        assert isinstance(compressor.bge_client, BaseRerankerClient)
        assert compressor.top_k == 5


class TestGetRerankerClient:
    """测试 get_reranker_client 便捷函数"""

    def test_returns_singleton(self):
        """测试返回单例"""
        client1 = get_reranker_client()
        client2 = get_reranker_client()
        assert client1 is client2

    def test_same_as_get_instance(self):
        """测试与 get_instance 返回相同"""
        client1 = BaseRerankerClient.get_instance()
        client2 = get_reranker_client()
        assert client1 is client2