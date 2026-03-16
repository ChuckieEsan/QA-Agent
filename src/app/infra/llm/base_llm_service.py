"""
默认 LLM 服务实现

使用 Provider 模式，支持多模型提供商和多模型。
继承 RunnableSerializable，符合 LangChain 最佳实践。

架构说明：
- 通过 provider 创建具体的 LangChain 模型实例
- 通过 model_name 指定使用该提供商的哪个模型
- 支持多提供商：deepseek/qwen/ollama
- 支持多模型：每个提供商可以有多个模型（generation/classification/optimization）
- 可以直接用于 LCEL 链：prompt | llm
"""

from typing import Optional, Type

from langchain_core.runnables import RunnableSerializable
from langchain_core.messages import BaseMessage
from src.app.infra.llm.providers import DeepSeekProvider, QwenProvider, OllamaProvider, BaseLLMProvider
from src.config.setting import settings
from pydantic import BaseModel, Field


class BaseLLMService(RunnableSerializable):
    """LangChain Runnable LLM 服务"""

    # 配置字段，用于 langchain 序列化
    model_name: str = Field(default="default", description="模型名称")

    def __init__(
        self,
        provider: Optional[BaseLLMProvider] = None,
        model_name: Optional[str] = None,
        provider_id: Optional[str] = None,
        **kwargs
    ):
        """
        初始化 LLM 服务

        Args:
            provider: 已创建的 Provider 实例，如不提供则根据 provider_id 创建
            model_name: 模型名称，如 "deepseek-chat"，如不提供则使用 Provider 默认
            provider_id: 提供商 ID，如 "deepseek"、"qwen"、"ollama"
        """
        # 先调用父类初始化，确保 Pydantic 字段先设置
        model_name = model_name or "default"
        super().__init__(model_name=model_name, **kwargs)

        # 如果没有提供 provider，根据 provider_id 创建
        if provider is None:
            provider_map = {
                "deepseek": DeepSeekProvider,
                "qwen": QwenProvider,
                "ollama": OllamaProvider,
            }

            pid = provider_id or settings.llm.default_provider
            provider_class = provider_map.get(pid)
            if not provider_class:
                raise ValueError(f"未知的提供商: {pid}")

            # 从配置中获取该提供商的配置
            provider_config = settings.llm.get_provider_config(pid)
            provider = provider_class(provider_config)

        # 使用 provider 创建具体的模型实例
        self._llm = provider.create_model(model_name)

    def invoke(self, input: list[BaseMessage], config=None):
        """同步调用"""
        return self._llm.invoke(input, config)

    async def ainvoke(self, input: list[BaseMessage], config=None):
        """异步调用"""
        return await self._llm.ainvoke(input, config)

    # 代理 langchain 模型的方法
    def with_structured_output(self, schema: Type[BaseModel], method: str = "function_calling"):
        """结构化输出 - 代理到底层 langchain model"""
        return self._llm.with_structured_output(schema, method=method)

    def bind(self, **kwargs):
        """绑定参数 - 代理到底层 langchain model"""
        return self._llm.bind(**kwargs)

    def bind_tools(self, *args, **kwargs):
        """绑定工具 - 代理到底层 langchain model"""
        return self._llm.bind_tools(*args, **kwargs)
    
