"""
LLM 提供商抽象基类
所有具体提供商必须实现此基类
"""

from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel
from src.config.setting import LLMProviderConfig


class BaseLLMProvider(ABC):
    """
    LLM 提供商抽象基类

    提供统一的接口来创建 LangChain ChatModel 实例。
    每个具体提供商（DeepSeek, Qwen, Ollama）必须实现 create_model 方法。
    """

    def __init__(self, config: LLMProviderConfig):
        """
        初始化提供商

        Args:
            config: 提供商配置，包含 api_key, base_url 等
        """
        self._config = config

    @property
    def config(self) -> LLMProviderConfig:
        """获取提供商配置"""
        return self._config

    @property
    def provider_id(self) -> str:
        """获取提供商 ID"""
        return self._config.provider_id

    @abstractmethod
    def create_model(self, model_name: Optional[str] = None, **kwargs) -> BaseChatModel:
        """
        创建 LangChain ChatModel 实例. 这里是考虑到一个模型提供商可能会提供多个不同的模型.
        因此需要手动指定创建的模型实例是什么

        Args:
            model_name: 模型名称，不传则使用配置中的默认模型
            **kwargs: 额外的模型参数（如 temperature, max_tokens 等）

        Returns:
            LangChain BaseChatModel 实例
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(provider_id={self.provider_id})"
