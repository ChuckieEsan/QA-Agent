"""
测试行政权力清单检索器
"""

import pytest
from src.app.components.retrievers.powers_retriever import PowersVectorRetriever


class TestPowersVectorRetriever:
    """测试行政权力清单检索器"""

    def test_retrieve_basic(self, sample_query):
        """测试基本检索功能"""
        context, results, metadata = PowersVectorRetriever().retrieve(sample_query, top_k=5)

        # 检查返回结构
        assert isinstance(context, str)
        assert isinstance(results, list)
        assert isinstance(metadata, dict)

        # 检查结果数量
        assert len(results) <= 5

        # 检查 metadata 中的 sources
        assert "sources" in metadata
        assert len(metadata["sources"]) == len(results)

        # 检查 metadata 的必要字段
        assert "query" in metadata
        assert "num_results" in metadata
        assert "avg_similarity" in metadata

    def test_retrieve_fields(self):
        """测试返回字段完整性"""
        context, results, metadata = PowersVectorRetriever().retrieve("公积金提取", top_k=3)

        if results:
            # 检查必要字段存在
            for result in results:
                assert "power_name" in result
                assert "department" in result
                assert "power_type" in result
                assert "similarity" in result
                assert "text" in result

                # 检查字段类型
                assert isinstance(result["power_name"], str)
                assert isinstance(result["department"], str)
                assert isinstance(result["power_type"], str)
                assert isinstance(result["similarity"], (int, float))

    def test_retrieve_similarity(self):
        """测试相似度"""
        context, results, metadata = PowersVectorRetriever().retrieve("行政许可", top_k=10)

        # 检查平均相似度
        if results:
            assert metadata["avg_similarity"] > 0

            # 检查每个结果的相似度
            for result in results:
                assert result["similarity"] >= 0

    def test_singleton(self):
        """测试单例模式"""
        retriever1 = PowersVectorRetriever()
        retriever2 = PowersVectorRetriever()
        assert retriever1 is retriever2

    def test_retrieve_no_results(self):
        """测试无结果情况"""
        # 使用一个不太可能匹配的查询
        context, results, metadata = PowersVectorRetriever().retrieve(
            "xyz123456789notfound", top_k=5
        )

        # 应该返回空结果或者有结果（取决于数据库内容）
        assert isinstance(context, str)
        assert isinstance(results, list)
        assert isinstance(metadata, dict)

    def test_retrieve_custom_top_k(self):
        """测试自定义 top_k"""
        context, results, metadata = PowersVectorRetriever().retrieve("处罚", top_k=3)
        assert len(results) <= 3

        context, results, metadata = PowersVectorRetriever().retrieve("处罚", top_k=10)
        assert len(results) <= 10

    def test_sources_structure(self):
        """测试 sources 结构"""
        context, results, metadata = PowersVectorRetriever().retrieve("公积金", top_k=3)

        if results:
            for source in metadata["sources"]:
                assert "rank" in source
                assert "power_name" in source
                assert "department" in source
                assert "power_type" in source
                assert "similarity" in source

    def test_retrieve_confidence(self):
        """测试置信度计算"""
        context, results, metadata = PowersVectorRetriever().retrieve("行政许可", top_k=5)

        retriever = PowersVectorRetriever()
        confidence = retriever.calculate_confidence(results)

        # 检查置信度范围
        assert 0.0 <= confidence <= 1.0