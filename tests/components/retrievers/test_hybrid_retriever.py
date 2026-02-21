"""
测试检索器组件
"""

import pytest
from src.app.components.retrievers.hybrid_retriever import (
    HybridVectorRetriever,
    retrieve_with_details,
    get_retriever_instance
)


class TestHybridVectorRetriever:
    """测试混合检索器"""

    def test_retrieve_with_details_basic(self, sample_query):
        """测试基本检索功能"""
        result = retrieve_with_details(sample_query, top_k=5)

        # 检查返回结构
        assert "query" in result
        assert "context" in result
        assert "sources" in result
        assert "metadata" in result

        # 检查结果数量
        assert result["num_sources"] > 0
        assert len(result["sources"]) <= 5

    def test_retrieve_with_details_similarity(self):
        """测试相似度阈值"""
        result = retrieve_with_details("雨露计划补贴标准", top_k=10)

        # 检查平均相似度
        assert result["metadata"]["avg_similarity"] > 0.5

        # 检查每个结果的相似度
        for source in result["sources"]:
            assert source["similarity"] >= result["metadata"]["threshold_applied"]

    def test_retrieve_with_details_metadata(self):
        """测试元数据完整性"""
        result = retrieve_with_details("公积金提取", top_k=3)

        # 检查元数据字段
        assert "query" in result["metadata"]
        assert "retrieval_time" in result["metadata"]
        assert "num_results" in result["metadata"]
        assert "avg_similarity" in result["metadata"]
        assert "threshold_applied" in result["metadata"]

    def test_get_retriever_instance_singleton(self):
        """测试单例模式"""
        retriever1 = get_retriever_instance()
        retriever2 = get_retriever_instance()

        assert retriever1 is retriever2

    def test_retriever_custom_config(self):
        """测试自定义配置（统一使用 min_similarity 命名）"""
        # 重置单例状态
        HybridVectorRetriever._instance = None
        HybridVectorRetriever._is_initialized = False

        config = {
            "top_k": 8,
            "cache_enabled": False,
            "min_similarity": 0.6
        }
        retriever = HybridVectorRetriever(config=config)

        # 检查配置生效
        assert retriever.default_top_k == 8
        assert retriever.cache_enabled == False
        assert retriever.min_similarity == 0.6

    def test_retrieve_with_details_confidence(self):
        """测试置信度计算"""
        result = retrieve_with_details("创业补贴政策", top_k=5)

        # 检查置信度字段
        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0

    def test_sources_composite_score(self):
        """测试综合评分"""
        result = retrieve_with_details("医保报销流程", top_k=5)

        # 检查综合评分字段
        for source in result["sources"]:
            assert "composite_score" in source
            assert 0.0 <= source["composite_score"] <= 1.0

    def test_threshold_filtering(self):
        """测试阈值过滤功能"""
        retriever = HybridVectorRetriever()

        # 使用较低阈值检索
        context, results, metadata = retriever.retrieve(
            "雨露计划",
            top_k=10
        )

        # 检查结果是否被过滤
        assert len(results) > 0

    def test_dynamic_threshold_adjustment(self):
        """测试动态阈值调整"""
        retriever = HybridVectorRetriever()

        # 使用模糊查询触发动态调整
        context, results, metadata = retriever.retrieve(
            "一些不明确的查询词语",
            top_k=5
        )

        # 应该有结果返回（动态调整生效）
        assert len(results) > 0


@pytest.mark.integration
class TestHybridVectorRetrieverIntegration:
    """集成测试 - 需要真实数据库"""

    @pytest.mark.skip(reason="需要真实数据库连接")
    def test_real_retrieval(self, sample_query):
        """测试真实检索"""
        result = retrieve_with_details(sample_query, top_k=5)
        assert result["num_sources"] > 0
