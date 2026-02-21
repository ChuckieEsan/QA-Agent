"""
测试 Agent 组件
"""

import pytest
import asyncio
from src import query_agentic_rag, RagAgent


class TestRagAgent:
    """测试 RAG Agent"""

    @pytest.mark.asyncio
    async def test_basic_query(self, sample_query):
        """测试基本查询"""
        result = await query_agentic_rag(query=sample_query)

        assert "answer" in result
        assert "classification" in result
        assert "sources" in result
        assert "metadata" in result

    @pytest.mark.asyncio
    async def test_with_history(self, sample_query, sample_history):
        """测试带历史的查询"""
        result = await query_agentic_rag(
            query="那什么时候能拿到补贴呢？",
            history=sample_history
        )

        assert "answer" in result

    @pytest.mark.asyncio
    async def test_classification_result(self, sample_query):
        """测试分类结果"""
        result = await query_agentic_rag(query=sample_query)

        classification = result["classification"]
        assert "type" in classification
        assert classification["type"] in ["advice", "complaint", "help", "consult"]


class TestRagAgentInstance:
    """测试 Agent 实例"""

    def test_agent_creation(self):
        """测试 Agent 创建"""
        agent = RagAgent()
        assert agent is not None
