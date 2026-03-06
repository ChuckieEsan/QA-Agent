from abc import ABC, abstractmethod
from typing import List, Optional, Union, TypeVar, Generic, Dict, Any, Type
from pydantic import BaseModel, Field, ConfigDict
from langchain_core.messages import BaseMessage
from src.app.infra.llm.providers.base_provider import BaseLLMProvider
from src.config.setting import LLMProviderConfig

# 定义消息类型的联合，支持字典或 LangChain 消息对象
MessageType = Union[Dict[str, str], BaseMessage]

# 泛型变量，表示结构化输出的类型
T = TypeVar('T', bound=BaseModel)

class BaseLLMService(ABC):
    """
    LLM 服务抽象基类

    提供统一的文本生成和结构化输出接口，支持同步和异步调用。
    所有具体实现必须实现抽象方法。

    支持通过 LLMConfig 配置模型参数，具体实现类应在 __init__ 中接收配置并初始化底层客户端。
    """

    def __init__(self, provider: BaseLLMProvider, **kwargs):
        """
        初始化 LLM 服务

        Args:
            config: 模型配置对象，包含连接参数和默认生成参数
            **kwargs: 额外的配置参数，可覆盖 config 中的值
        """
        self._provider = provider

    # 通用文本生成
    @abstractmethod
    def generate(
        self,
        messages: List[MessageType],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        生成文本响应（同步）

        Args:
            messages: 消息列表，每条可以是 {"role": "user", "content": "..."} 或 BaseMessage 对象
            temperature: 采样温度，默认 None 使用模型默认值
            max_tokens: 最大生成 token 数
            top_p: 核采样参数
            **kwargs: 其他模型参数

        Returns:
            生成的文本内容
        """
        pass

    @abstractmethod
    async def agenerate(
        self,
        messages: List[MessageType],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> str:
        """异步版本 generate"""
        pass

    # 结构化输出
    @abstractmethod
    def generate_structured(
        self,
        messages: List[MessageType],
        response_model: Type[T],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> T:
        """
        生成符合 Pydantic 模型的结构化输出（同步）

        底层使用函数调用或 response_format 强制模型返回 JSON，并自动解析为模型实例。

        Args:
            messages: 消息列表
            response_model: Pydantic 模型类，用于定义输出格式
            temperature: 采样温度
            max_tokens: 最大生成 token 数
            **kwargs: 其他参数

        Returns:
            response_model 的实例，已通过 Pydantic 验证
        """
        pass

    @abstractmethod
    async def agenerate_structured(
        self,
        messages: List[MessageType],
        response_model: Type[T],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> T:
        """异步版本 generate_structured"""
        pass