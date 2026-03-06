"""
DeepSeek 提供商实现
"""

from typing import Optional
from langchain_openai import ChatOpenAI
from src.app.infra.llm.providers.base_provider import BaseLLMProvider
from src.config.setting import LLMProviderConfig


class DeepSeekProvider(BaseLLMProvider):
    """
    DeepSeek 模型提供商

    使用 OpenAI 兼容接口接入 DeepSeek API
    文档：https://platform.deepseek.com/api-docs/
    """

    def __init__(self, config: LLMProviderConfig):
        super().__init__(config)

    def create_model(self, model_name: Optional[str] = None, **kwargs) -> ChatOpenAI:
        """
        创建 DeepSeek ChatModel 实例

        Args:
            model_name: 模型名称，如 "deepseek-chat", "deepseek-coder"
            **kwargs: 额外的模型参数

        Returns:
            ChatOpenAI 实例，配置为使用 DeepSeek API
        """
        name = model_name or self._config.models.get("generation", "deepseek-chat")

        return ChatOpenAI(
            api_key=self._config.api_key,
            base_url=self._config.base_url or "https://api.deepseek.com",
            model=name,
            **kwargs
        )
