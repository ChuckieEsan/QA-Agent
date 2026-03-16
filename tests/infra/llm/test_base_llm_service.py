"""
BaseLLMService 测试类

使用真实 API 测试 BaseLLMService 的各种功能
"""

import pytest
from pydantic import BaseModel
from typing import Literal
from langchain_core.messages import HumanMessage
from src.app.infra.llm import BaseLLMService, create_llm_service
from langchain_core.prompts import ChatPromptTemplate


class TestBaseLLMService:
    """BaseLLMService 测试类"""

    @pytest.fixture
    def llm_deepseek(self):
        """创建 DeepSeek LLM 服务实例"""
        return create_llm_service(
            provider_id="deepseek",
            model_name="deepseek-chat"
        )

    @pytest.fixture
    def llm_qwen(self):
        """创建 Qwen LLM 服务实例"""
        return create_llm_service(
            provider_id="qwen",
            model_name="qwen-turbo"
        )

    def test_create_llm_service(self):
        """测试创建 LLM 服务"""
        llm = create_llm_service(
            provider_id="deepseek",
            model_name="deepseek-chat"
        )
        assert llm is not None
        assert llm.model_name == "deepseek-chat"

    def test_invoke(self, llm_deepseek):
        """测试同步调用"""

        result = llm_deepseek.invoke(
            [HumanMessage(content="你好，请用一句话介绍自己")],
            config={"temperature": 0.5, "max_tokens": 100}
        )
        assert result is not None
        assert len(result.content) > 0

    @pytest.mark.asyncio
    async def test_ainvoke(self, llm_deepseek):
        """测试异步调用"""
    
        result = await llm_deepseek.ainvoke(
            [HumanMessage(content="你好，请用一句话介绍自己")],
            config={"temperature": 0.5, "max_tokens": 100}
        )
        assert result is not None
        assert len(result.content) > 0

    def test_with_structured_output(self, llm_deepseek):
        """测试结构化输出"""
        class AnswerFormat(BaseModel):
            answer: str
            confidence: float

        chain = llm_deepseek.with_structured_output(AnswerFormat)

        result = chain.invoke(
            [HumanMessage(content="请回答：1+1等于几？只返回答案")],
            config={"temperature": 0}
        )

        assert result is not None
        assert hasattr(result, "answer")
        assert hasattr(result, "confidence")

    def test_bind_temperature(self, llm_deepseek):
        """测试绑定参数 - temperature"""
    
        # 绑定 temperature
        bound_llm = llm_deepseek.bind(temperature=0.9)

        result = bound_llm.invoke(
            [HumanMessage(content="写一个关于春天的短句")]
        )

        assert result is not None
        assert len(result.content) > 0

    def test_lcel_chain(self, llm_deepseek):
        """测试 LCEL 链式调用"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个助手，请用{style}风格回答用户问题。"),
            ("user", "{question}")
        ])

        chain = prompt | llm_deepseek

        result = chain.invoke({
            "style": "简洁",
            "question": "什么是人工智能？"
        })

        assert result is not None
        assert len(result.content) > 0


class TestDeepSeekProvider:
    """DeepSeek 提供商测试"""

    def test_deepseek_chat(self):
        """测试 DeepSeek Chat 模型"""
        llm = create_llm_service(
            provider_id="deepseek",
            model_name="deepseek-chat"
        )

        result = llm.invoke(
            [HumanMessage(content="你好")],
            config={"temperature": 0.5}
        )

        assert result is not None
        print(f"DeepSeek response: {result.content}")


class TestQwenProvider:
    """Qwen 提供商测试"""

    @pytest.mark.skipif(
        True,  # 默认跳过，需要手动启用
        reason="需要配置 Qwen API"
    )
    def test_qwen_chat(self):
        """测试 Qwen Chat 模型"""
        llm = create_llm_service(
            provider_id="qwen",
            model_name="qwen-turbo"
        )

        result = llm.invoke(
            [HumanMessage(content="你好")],
            config={"temperature": 0.5}
        )

        assert result is not None
        print(f"Qwen response: {result.content}")