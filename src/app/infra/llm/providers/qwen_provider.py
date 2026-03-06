"""
Qwen (通义千问) 提供商实现

使用 OpenAI 兼容接口接入 DashScope API
文档：https://help.aliyun.com/zh/dashscope/developer-reference/compatibility-of-openai-api-with-dashscope/
"""

from typing import Optional
from langchain_openai import ChatOpenAI
from src.app.infra.llm.providers.base_provider import BaseLLMProvider
from src.config.setting import LLMProviderConfig


class QwenProvider(BaseLLMProvider):
    """
    Qwen (通义千问) 模型提供商

    使用 OpenAI 兼容接口接入 DashScope API
    """

    def __init__(self, config: LLMProviderConfig):
        super().__init__(config)

    def create_model(self, model_name: Optional[str] = None, **kwargs) -> ChatOpenAI:
        """
        创建 Qwen ChatModel 实例

        Args:
            model_name: 模型名称，如 "qwen-max", "qwen-plus", "qwen-turbo"
            **kwargs: 额外的模型参数

        Returns:
            ChatOpenAI 实例，配置为使用 DashScope OpenAI 兼容接口
        """
        name = model_name or self._config.models.get("generation", "qwen-max")

        # DashScope OpenAI 兼容接口
        # base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
        base_url = self._config.base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"

        return ChatOpenAI(
            api_key=self._config.api_key,
            base_url=base_url,
            model=name,
            **kwargs
        )
