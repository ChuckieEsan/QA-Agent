"""
生成器抽象基类
提供统一的文本生成接口，使用 Message 列表作为入参
"""
import copy
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, AsyncGenerator, Union, Any

from src.app.infra.llm.schema import Message, DEFAULT_SYSTEM_MESSAGE


class BaseGenerator(ABC):
    """
    生成器抽象基类

    所有生成器实现都应该继承此类
    使用 Message 列表进行交互，支持：
    - system/user/assistant/function 角色
    - 多模态内容（text/image/file/audio/video）
    - function_call 格式
    """

    def __init__(self, system_message: Optional[str] = None):
        """
        初始化生成器

        Args:
            system_message: 系统消息（可选）
        """
        self.system_message = system_message or DEFAULT_SYSTEM_MESSAGE

    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> str:
        """
        同步生成文本（使用主模型）

        Args:
            messages: 消息列表，每个消息包含:
                - role: "system" | "user" | "assistant" | "function"
                - content: str | List[ContentItem]
                - name: Optional[str]
                - function_call: Optional[dict]
            **kwargs: 其他生成参数（temperature, max_tokens, top_p 等）

        Returns:
            生成的文本
        """

    @abstractmethod
    async def generate_stream(
        self,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        流式生成文本（使用主模型）

        Args:
            messages: 消息列表
            **kwargs: 其他生成参数

        Yields:
            生成的文本片段
        """

    @abstractmethod
    async def generate_with_validation(
        self,
        messages: List[Dict[str, Any]],
        validation_criteria: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        生成并验证文本（使用主模型）

        Args:
            messages: 消息列表
            validation_criteria: 验证标准
            **kwargs: 其他生成参数

        Returns:
            {
                "text": str,
                "quality_score": float,
                "passed_validation": bool,
                "validation_details": Dict
            }
        """

    @abstractmethod
    async def initialize(self) -> None:
        """
        初始化生成器资源
        """

    # ==================== 工具方法 ====================

    def _prepare_messages(
        self,
        messages: List[Dict[str, Any]],
        system_message: Optional[str] = None,
        append_system: bool = True
    ) -> List[Message]:
        """
        准备消息列表，支持多种输入格式

        Args:
            messages: 输入消息（dict 或 Message 列表）
            system_message: 系统消息（可选）
            append_system: 是否在开头添加系统消息

        Returns:
            Message 列表
        """
        # 深拷贝避免修改原始消息
        messages = copy.deepcopy(messages)

        # 转换为 Message 列表
        message_list: List[Message] = []
        for msg in messages:
            if isinstance(msg, dict):
                message_list.append(Message(**msg))
            else:
                message_list.append(msg)

        # 添加系统消息
        if system_message or self.system_message:
            sys_content = system_message or self.system_message
            if append_system:
                # 检查是否已存在系统消息
                if not message_list or message_list[0].role != "system":
                    message_list.insert(0, Message(
                        role="system",
                        content=sys_content or DEFAULT_SYSTEM_MESSAGE
                    ))
                else:
                    # 已有系统消息，追加内容
                    if isinstance(message_list[0].content, str):
                        message_list[0].content = sys_content + "\n\n" + message_list[0].content

        return message_list

    def _build_prompt_from_messages(
        self,
        messages: List[Message],
        add_upload_info: bool = False
    ) -> str:
        """
        从 Message 列表构建 Prompt 字符串

        Args:
            messages: Message 列表
            add_upload_info: 是否添加上传信息

        Returns:
            Prompt 字符串
        """
        return self._convert_to_prompt(messages)

    def _convert_to_prompt(self, messages: List[Message]) -> str:
        """
        将 Message 列表转换为 Prompt 字符串（格式化为对话形式）

        Args:
            messages: Message 列表

        Returns:
            Prompt 字符串
        """
        parts = []

        for msg in messages:
            role = msg.role
            content = msg.content

            if isinstance(content, list):
                # 提取文本内容
                text_parts = []
                for item in content:
                    if hasattr(item, 'text') and item.text:
                        text_parts.append(item.text)
                    elif isinstance(item, dict) and 'text' in item:
                        text_parts.append(item['text'])
                content = "\n".join(text_parts)

            if role == "system":
                parts.append(f"## 系统指令\n{content}")
            elif role == "user":
                parts.append(f"## 用户查询\n{content}")
            elif role == "assistant":
                parts.append(f"## 助手回复\n{content}")
            elif role == "function":
                parts.append(f"## 工具结果\n{content}")

        return "\n\n".join(parts)

    def _add_assistant_message(
        self,
        messages: List[Message],
        content: str
    ) -> List[Message]:
        """
        添加助手消息到消息列表

        Args:
            messages: 消息列表
            content: 助手消息内容

        Returns:
            更新后的消息列表
        """
        messages.append(Message(role="assistant", content=content))
        return messages

    def _add_user_message(
        self,
        messages: List[Message],
        content: str
    ) -> List[Message]:
        """
        添加用户消息到消息列表

        Args:
            messages: 消息列表
            content: 用户消息内容

        Returns:
            更新后的消息列表
        """
        messages.append(Message(role="user", content=content))
        return messages

    def _add_function_message(
        self,
        messages: List[Message],
        name: str,
        content: str
    ) -> List[Message]:
        """
        添加函数消息到消息列表

        Args:
            messages: 消息列表
            name: 函数名称
            content: 函数结果

        Returns:
            更新后的消息列表
        """
        messages.append(Message(role="function", name=name, content=content))
        return messages
