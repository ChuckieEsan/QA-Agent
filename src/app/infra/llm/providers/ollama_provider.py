"""
Ollama (本地部署) 提供商实现

使用 OpenAI 兼容接口接入 Ollama API
文档：https://github.com/ollama/ollama/blob/main/docs/openai.md
"""

from typing import Optional
from langchain_openai import ChatOpenAI
from src.app.infra.llm.providers.base_provider import BaseLLMProvider
from src.config.setting import LLMProviderConfig


class OllamaProvider(BaseLLMProvider):
    """
    Ollama 本地模型提供商

    使用 OpenAI 兼容接口接入本地 Ollama 服务
    """

    def __init__(self, config: LLMProviderConfig):
        super().__init__(config)

    def create_model(self, model_name: Optional[str] = None, **kwargs) -> ChatOpenAI:
        """
        创建 Ollama ChatModel 实例

        Args:
            model_name: 模型名称，如 "qwen2.5:7b", "llama3:8b"
            **kwargs: 额外的模型参数

        Returns:
            ChatOpenAI 实例，配置为使用 Ollama OpenAI 兼容接口
        """
        name = model_name or self._config.models.get("generation", "qwen2.5:7b")

        # Ollama OpenAI 兼容接口
        # base_url: http://localhost:11434/v1
        base_url = self._config.base_url or "http://localhost:11434/v1"

        # Ollama 通常不需要 API Key，但 OpenAI 客户端要求传入
        # 传入任意非空字符串即可
        api_key = self._config.api_key or "ollama"

        return ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=name,
            **kwargs
        )
