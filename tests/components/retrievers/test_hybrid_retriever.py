"""
测试检索器组件
"""

import pytest
from src.app.components.retrievers.hybrid_retriever import HybridVectorRetriever


class TestHybridVectorRetriever:
    """测试混合检索器"""

    def test_retrieve_basic(self, sample_query):
        """测试基本检索功能"""
        context, results, metadata = HybridVectorRetriever().retrieve(sample_query, top_k=5)

        # 检查返回结构
        assert isinstance(context, str)
        assert isinstance(results, list)
        assert isinstance(metadata, dict)

        # 检查结果数量（可能为0，取决于数据库状态）
        assert len(results) <= 5

        # 检查 metadata 中的 sources
        assert "sources" in metadata
        assert len(metadata["sources"]) == len(results)

        # 检查 metadata 的必要字段
        assert "query" in metadata
        assert "num_results" in metadata
        assert "avg_similarity" in metadata

    def test_retrieve_similarity(self):
        """测试相似度阈值"""
        context, results, metadata = HybridVectorRetriever().retrieve("雨露计划补贴标准", top_k=10)

        # 检查平均相似度
        assert metadata["avg_similarity"] > 0.5

        # 检查每个结果的相似度
        for result in results:
            assert result["similarity"] >= metadata["threshold_applied"]

    def test_retrieve_metadata(self):
        """测试元数据完整性"""
        context, results, metadata = HybridVectorRetriever().retrieve("公积金提取", top_k=3)

        # 检查元数据字段
        assert "query" in metadata
        assert "retrieval_time" in metadata
        assert "num_results" in metadata
        assert "avg_similarity" in metadata
        assert "threshold_applied" in metadata

    def test_singleton(self):
        """测试单例模式"""
        retriever1 = HybridVectorRetriever()
        retriever2 = HybridVectorRetriever()
        assert retriever1 is retriever2

    def test_custom_config(self):
        """测试自定义配置"""
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

    def test_retrieve_confidence(self):
        """测试置信度计算"""
        context, results, metadata = HybridVectorRetriever().retrieve("创业补贴政策", top_k=5)

        # 从 results 计算置信度
        retriever = HybridVectorRetriever()
        confidence = retriever.calculate_confidence(results)

        # 检查置信度字段
        assert 0.0 <= confidence <= 1.0

    def test_sources_composite_score(self):
        """测试综合评分"""
        context, results, metadata = HybridVectorRetriever().retrieve("医保报销流程", top_k=5)

        # 检查综合评分字段存在且为有效数值
        for result in results:
            assert "composite_score" in result
            assert isinstance(result["composite_score"], (int, float)), "composite_score 应该是数值类型"

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
            "最近噪音实在是太大了",
            top_k=5
        )

        # 应该有结果返回（动态调整生效）
        assert len(results) > 0

    def test_bge_reranker_working(self):
        """测试 BGE 重排模型正常工作"""
        context, results, metadata = HybridVectorRetriever().retrieve("雨露计划补贴", top_k=5)

        if results:
            # 检查重排得分字段
            for result in results:
                assert "rerank_score" in result, "结果应包含 rerank_score 字段"
                assert "composite_score" in result, "结果应包含 composite_score 字段"

    def test_title_and_department_fields(self):
        """测试 title 和 department 字段存在"""
        context, results, metadata = HybridVectorRetriever().retrieve("公积金提取", top_k=3)

        if results:
            # 检查 title 和 department 字段
            for result in results:
                assert "title" in result, "结果应包含 title 字段"
                assert "department" in result, "结果应包含 department 字段"
                assert isinstance(result["title"], str), "title 应该是字符串类型"
                assert isinstance(result["department"], str), "department 应该是字符串类型"

    def test_retrieve_without_cache(self):
        """测试禁用缓存的检索"""
        # 重置单例状态
        HybridVectorRetriever._instance = None
        HybridVectorRetriever._is_initialized = False

        # 创建禁用缓存的 retriever
        config = {
            "cache_enabled": False,
            "top_k": 5
        }
        retriever = HybridVectorRetriever(config=config)

        # 验证缓存已禁用
        assert retriever.cache_enabled == False

        # 执行检索
        context, results, metadata = retriever.retrieve("医保报销流程", top_k=5)

        # 检查返回结构
        assert isinstance(context, str)
        assert isinstance(results, list)
        assert isinstance(metadata, dict)
        
        