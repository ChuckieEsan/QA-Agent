"""
默认 LLM 服务实现

使用 Provider 模式，支持多模型提供商和多模型。

架构说明：
- 通过 provider 创建具体的 LangChain 模型实例
- 通过 model_name 指定使用该提供商的哪个模型
- 支持多提供商：deepseek/qwen/ollama
- 支持多模型：每个提供商可以有多个模型（generation/classification/optimization）
"""

from .base_llm_service import BaseLLMService, MessageType, T
from typing import List, Optional, Type
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from src.app.infra.llm.providers import DeepSeekProvider, QwenProvider, OllamaProvider, BaseLLMProvider
from src.config.setting import settings

class DefaultLLMService(BaseLLMService):
    """默认 LLM 服务实现"""

    def __init__(
        self,
        provider: Optional[BaseLLMProvider] = None,
        model_name: Optional[str] = None,
        provider_id: Optional[str] = None,
    ):
        """
        初始化 LLM 服务

        Args:
            provider: 已创建的 Provider 实例，如不提供则根据 provider_id 创建
            model_name: 模型名称，如 "deepseek-chat"，如不提供则使用 Provider 默认
            provider_id: 提供商 ID，如 "deepseek"、"qwen"、"ollama"
        """
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

        # 调用父类初始化
        super().__init__(provider)

        # 使用 provider 创建具体的模型实例
        self._llm = provider.create_model(model_name)

        # 保存模型名称供后续使用
        self._model_name = model_name or "default"

    def _convert_messages(self, messages: List[MessageType]) -> List[BaseMessage]:
        """将输入消息转换为 LangChain BaseMessage 列表"""
        result = []
        for msg in messages:
            if isinstance(msg, BaseMessage):
                result.append(msg)
            elif isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    result.append(SystemMessage(content=content))
                else:
                    result.append(HumanMessage(content=content))
            else:
                raise TypeError(f"Unsupported message type: {type(msg)}")
        return result

    def generate(
        self,
        messages: List[MessageType],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> str:
        """同步生成文本"""
        langchain_messages = self._convert_messages(messages)
        response = self._llm.invoke(
            langchain_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs
        )
        return response.content

    async def agenerate(
        self,
        messages: List[MessageType],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> str:
        """异步生成文本"""
        langchain_messages = self._convert_messages(messages)
        response = await self._llm.ainvoke(
            langchain_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs
        )
        return response.content

    def generate_structured(
        self,
        messages: List[MessageType],
        response_model: Type[T],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> T:
        """同步生成结构化输出"""
        structured_llm = self._llm.with_structured_output(response_model, method="function_calling")
        langchain_messages = self._convert_messages(messages)
        result = structured_llm.invoke(
            langchain_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        return result

    async def agenerate_structured(
        self,
        messages: List[MessageType],
        response_model: Type[T],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> T:
        """异步生成结构化输出"""
        structured_llm = self._llm.with_structured_output(response_model, method="function_calling")
        langchain_messages = self._convert_messages(messages)
        result = await structured_llm.ainvoke(
            langchain_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        return result
