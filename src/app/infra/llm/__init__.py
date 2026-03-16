"""
LLM 基础设施模块
提供多模型、多提供商支持
"""

from .base_llm_service import BaseLLMService
from .providers import BaseLLMProvider, DeepSeekProvider, QwenProvider, OllamaProvider
from typing import Optional


__all__ = [
    "BaseLLMService",
    "MessageType",
    "BaseLLMProvider",
    "DeepSeekProvider",
    "QwenProvider",
    "OllamaProvider",
    "create_llm_service",
]


def create_llm_service(
    provider_id: Optional[str] = None,
    model_name: Optional[str] = None,
) -> BaseLLMService:
    """
    创建 LLM 服务的便捷工厂函数

    使用示例：
        ### 使用默认提供商的默认模型
        llm = create_llm_service()

        ### 使用指定提供商
        llm = create_llm_service(provider_id="qwen")

        ### 使用指定提供商的指定模型
        llm = create_llm_service(provider_id="deepseek", model_name="deepseek-chat")

        ### 使用 Ollama 的 72b 模型
        llm = create_llm_service(provider_id="ollama", model_name="qwen2.5:72b")
    """
    return BaseLLMService(provider_id=provider_id, model_name=model_name)
