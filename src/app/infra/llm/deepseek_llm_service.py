from base_llm_service import BaseLLMService, MessageType, T
from typing import List, Optional, Union, Dict, Any, Type
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel
import json

class DeepSeekLLMService(BaseLLMService):
    def __init__(self, api_key: str, model: str = "deepseek-chat", base_url: str = "https://api.deepseek.com/v1"):
        self.llm = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=0  # 默认值，可在调用时覆盖
        )

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
        langchain_messages = self._convert_messages(messages)
        response = self.llm.invoke(
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
        langchain_messages = self._convert_messages(messages)
        response = await self.llm.ainvoke(
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
        structured_llm = self.llm.with_structured_output(response_model)
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
        structured_llm = self.llm.with_structured_output(response_model)
        langchain_messages = self._convert_messages(messages)
        result = await structured_llm.ainvoke(
            langchain_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        return result
    
    