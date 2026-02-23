"""
Agent 抽象基类
定义统一的 Agent 接口，封装通用的 LLM 调用、工具执行等逻辑
"""

import json
import traceback
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from src.app.infra.llm.multi_model_service import get_optimizer_llm_service
from src.app.infra.utils.logger import get_logger
from src.app.infra.llm.schema import Message, FUNCTION, SYSTEM, USER
from dashscope import Generation
from src.app.agents.tools import BaseTool, ToolRegistry
from src.config import settings

logger = get_logger(__name__)


class BaseAgent(ABC):
    """
    通用 Agent 基类，基于 Qwen Agent 的 Agent 类设计

    封装了通用的：
    - LLM 调用逻辑
    - 工具执行逻辑
    - 参数验证逻辑
    - 结果格式化逻辑
    - 消息历史管理

    子类只需实现：
    - _build_system_prompt() - 构建系统提示
    - _build_tools_description() - 构建工具描述
    - _run_cycle() - 核心运行循环
    """

    def __init__(
        self,
        name: str = "Agent",
        tools: Dict[str, BaseTool] = None,
        llm_service: Optional[Any] = None,
        max_retries: int = None,
        verbose: bool = False,
    ):
        """
        初始化 Agent

        Args:
            name: Agent 名称
            llm_service: LLM 服务实例（可选，默认使用 get_optimizer_llm_service）
            max_retries: 工具执行最大重试次数
        """
        self.name = name
        self.llm_service = llm_service or get_optimizer_llm_service()
        self.max_retries = (
            max_retries or settings.llm.max_function_call_retries
        )
        self.tools = tools or ToolRegistry.list_all()
        self.messages: List[Message] = []
        self._initialized = False
        self.verbose = verbose

        logger.info(f"✅ {self.__class__.__name__} 初始化完成")

    async def initialize(self) -> None:
        """
        初始化 Agent 资源

        子类可以重写此方法，用于：
        - 初始化依赖组件
        - 预热模型
        - 加载配置
        """
        logger.info(f"⏳ [{self.__class__.__name__}] 初始化工具...")
        for tool_name, tool in self.tools.items():
            if hasattr(tool, "initialize") and callable(tool.initialize):
                try:
                    await tool.initialize()
                    logger.info(f"  ✅ {tool_name} 已预热")
                except Exception as e:
                    logger.warning(f"  ⚠️ {tool_name} 预热失败: {e}")
        self._initialized = True
        logger.info(f"✅ [{self.__class__.__name__}] 初始化完成")

    def get_status(self) -> Dict[str, Any]:
        """
        获取 Agent 状态

        Returns:
            状态信息字典
        """
        return {
            "initialized": self._initialized,
            "name": self.name,
            "max_retries": self.max_retries,
            "tools": list(self.tools.keys()),
            "verbose": self.verbose,
            "message_count": len(self.messages),
        }

    async def _call_llm(
        self, messages: List[Message], system_prompt: Optional[str] = None
    ) -> List[Message]:
        """
        与 LLM 交互，返回更新后的消息列表

        Args:
            messages: 当前消息列表
            system_prompt: 系统提示（可选，如果提供则替换 messages 中的 system 消息）

        Returns:
            更新后的消息列表（包含 LLM 响应）
        """
        try:
            # 如果提供了 system_prompt，更新消息列表
            if system_prompt is not None:
                messages = self._update_system_message(messages, system_prompt)

            # 转换 Message 为 dict 列表
            messages_dict = [msg.model_dump() for msg in messages]

            # 调用 LLM
            response = Generation.call(
                model=self.llm_service.get_model_name(),
                messages=messages_dict,
                temperature=self.llm_service.get_config().temperature,
                max_tokens=self.llm_service.get_config().max_tokens,
                top_p=self.llm_service.get_config().top_p,
                result_format="message",
            )

            if response.status_code == 200:
                # 解析 LLM 响应
                choice = response.output.choices[0]
                response_msg = choice.get("message", {})

                # 构建 Message 对象
                msg = Message(
                    role=response_msg.get("role", "assistant"),
                    content=response_msg.get("content", ""),
                    function_call=response_msg.get("function_call"),
                )
                messages.append(msg)

            return messages

        except Exception as e:
            logger.error(f"❌ LLM 调用失败: {e}")
            logger.error(traceback.format_exc())
            raise

    def _update_system_message(
        self, messages: List[Message], system_prompt: str
    ) -> List[Message]:
        """
        更新消息列表中的 system 消息

        Args:
            messages: 消息列表
            system_prompt: 新的系统提示

        Returns:
            更新后的消息列表
        """
        messages = messages.copy()

        # 如果已有 system 消息，更新它
        for i, msg in enumerate(messages):
            if msg.role == SYSTEM:
                messages[i] = Message(role=SYSTEM, content=system_prompt)
                return messages

        # 否则添加 system 消息
        messages.insert(0, Message(role=SYSTEM, content=system_prompt))
        return messages

    # ==================== 工具调用检测 ====================

    def _detect_tool(self, message: Message) -> Tuple[bool, str, Dict[str, Any]]:
        """
        检测消息中的工具调用，支持两种格式：
        1. function_call 格式（OpenAI 兼容）
        2. 文本格式（ReAct 风格）

        Args:
            message: Message 对象

        Returns:
            (是否有工具调用, 工具名称, 工具参数)
        """
        # 格式1: function_call 格式
        if message.function_call:
            func_call = message.function_call
            func_name = func_call.name
            func_args = func_call.arguments
            try:
                args_dict = json.loads(func_args)
            except json.JSONDecodeError:
                args_dict = {"query": func_args}
            return True, func_name, args_dict

        # 格式2: 文本格式
        text = message.content or ""
        if isinstance(text, list):
            text = "".join(item.value for item in text if hasattr(item, "value"))

        # 解析 Thought/Action/Action Input
        special_func_token = "\nAction:"
        special_args_token = "\nAction Input:"
        func_name, func_args = None, None

        i = text.rfind(special_func_token)
        j = text.rfind(special_args_token)

        if 0 <= i < j:
            func_name = text[i + len(special_func_token) : j].strip()
            func_args = text[j + len(special_args_token) :].strip()
            try:
                func_args = json.loads(func_args)
            except json.JSONDecodeError:
                func_args = {"query": func_args}
            return True, func_name, func_args

        return False, "", {}

    # ==================== 工具执行 ====================

    async def _execute_tool(
        self,
        action: str,
        action_input: Dict[str, Any],
        max_retries: Optional[int] = None,
    ) -> Tuple[str, float]:
        """
        执行工具并格式化结果（带重试机制）

        Args:
            action: 动作名称
            action_input: 动作参数
            max_retries: 最大重试次数（可选，默认使用 self.max_retries）

        Returns:
            (observation, execution_time)
        """
        start_time = datetime.now()
        max_retries = max_retries if max_retries is not None else self.max_retries

        # 1. 验证参数
        is_valid, error_msg = self._validate_tool_input(action, action_input)
        if not is_valid:
            return f"错误：{error_msg}", 0.0

        if action == "Final Answer":
            answer = action_input.get("answer", "")
            return answer, 0.0

        if action not in self.tools:
            return f"错误：未知工具 {action}", 0.0

        # 2. 执行工具
        tool = self.tools[action]
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                result = await tool.execute(**action_input)

                # 格式化观察结果
                observation = self._format_observation(result)

                return observation, (datetime.now() - start_time).total_seconds()

            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    import asyncio

                    await asyncio.sleep(1.0 * (attempt + 1))

        # 所有重试失败
        logger.error(f"❌ 工具执行最终失败: {last_error}")
        return f"错误：{str(last_error)}", (datetime.now() - start_time).total_seconds()

    def _validate_tool_input(
        self, action: str, action_input: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        验证工具执行参数是否合法

        Args:
            action: 动作名称
            action_input: 动作参数

        Returns:
            (is_valid, error_message)
        """
        if action == "Final Answer":
            return True, ""

        if action not in self.tools:
            return False, f"未知工具: {action}"

        # 检查必要参数
        tool = self.tools[action]
        required_params = tool.get_required_parameters()

        for param in required_params:
            if param not in action_input:
                return False, f"缺少必要参数: {param}"

        return True, ""

    def _format_observation(self, result: Dict[str, Any]) -> str:
        """
        格式化工具执行结果为字符串
        TODO: 现阶段为直接使用 json.dumps

        Args:
            result: 工具执行结果

        Returns:
            格式化后的观察文本
        """
        return json.dumps(result, ensure_ascii=False)

    def _build_tools_description(self) -> str:
        """
        构建工具描述

        Returns:
            工具描述文本
        """
        tools_info_parts = []
        for tool_name, tool in self.tools.items():
            schema = tool.get_schema()
            tool_desc = f"## {tool_name}\n描述：{tool.description}"
            if "parameters" in schema:
                tool_desc += "\n参数说明："
                for param, desc in schema["parameters"].items():
                    tool_desc += f"\n- {param}: {desc}"

            tools_info_parts.append(tool_desc)

        return "\n\n".join(tools_info_parts)

    # ==================== 抽象方法 ====================

    @abstractmethod
    async def run(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        处理查询的核心方法

        Args:
            query: 用户查询
            **kwargs: 其他参数

        Returns:
            处理结果字典
        """
        pass

    @abstractmethod
    def _build_system_prompt(self) -> str:
        """
        构建系统提示（子类实现）

        Returns:
            系统提示文本
        """
        pass


if __name__ == "__main__":
    class TestAgent(BaseAgent):
        async def run(self, query: str, **kwargs) -> Dict[str, Any]:
            pass
        
        def _build_system_prompt(self) -> str:
            return "你是一个有用的助手"
    agent = TestAgent()
    import asyncio
    asyncio.run(agent._call_llm([Message(role=USER, content="1+1等于几")]))