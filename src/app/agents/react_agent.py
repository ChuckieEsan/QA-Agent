"""
ReAct Agent 核心实现
Thought-Action-Observation 循环引擎

ReAct范式：
1. Thought: LLM 分析当前状态并生成思考
2. Action: 选择并执行工具
3. Observation: 获取工具执行结果
4. 循环直到生成最终答案或达到最大步数

使用 Message 列表维护对话历史，降低 token 消耗：
- system_prompt 只在开始时设置一次
- 工具结果以 Message 形式添加到历史中
- 避免重复传递大量示例

使用装饰器模式注册工具：
    from src.app.agents.tools.registry import ToolRegistry

    @ToolRegistry.register()
    class MyTool(BaseTool):
        name = "my_tool"
        description = "我的工具"

        async def execute(self, **kwargs) -> dict:
            return {"result": "xxx"}

使用 ReactAgent：
    from src.app.agents import ReactAgent
    from src.app.agents.tools import ToolRegistry

    tools = {
        "retrieve": ToolRegistry.get_instance("retrieve"),
        "generate": ToolRegistry.get_instance("generate"),
        "classify": ToolRegistry.get_instance("classify"),
        "validate": ToolRegistry.get_instance("validate"),
    }
    agent = ReactAgent(tools, max_steps=5)
    result = await agent.process("2024年泸州雨露计划补贴标准")
    print(result["answer"])
    # 或查看完整消息历史
    for msg in result["messages"]:
        print(f"{msg['role']}: {msg['content'][:50]}...")
"""

import json
import traceback
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from src.app.infra.llm.multi_model_service import (
    get_optimizer_llm_service,
    get_heavy_llm_service
)
from src.app.infra.utils.logger import get_logger
from src.app.infra.llm.schema import Message, FUNCTION, SYSTEM, USER
from dashscope import Generation

from src.app.agents.tools import BaseTool
from src.config.setting import settings

logger = get_logger(__name__)


# ==================== 常量定义 ====================

# 工具执行前参数验证规则
TOOL_PARAMETER_REQUIREMENTS = {
    "retrieve": {
        "required": ["query"],
        "defaults": {"top_k": 5, "threshold": 0.5}
    },
    "generate": {
        "required": ["prompt"],
        "defaults": {"context": "", "history": None}
    },
    "classify": {
        "required": ["query"],
        "defaults": {}
    },
    "validate": {
        "required": ["answer", "query"],
        "defaults": {"context": ""}
    }
}


class ReactAgent:
    """
    ReAct Agent 核心类

    实现 Thought-Action-Observation 循环：
    1. Thought: LLM 分析当前状态并生成思考
    2. Action: 选择并执行工具
    3. Observation: 获取工具执行结果
    4. 循环直到生成最终答案或达到最大步数

    使用 Message 数组维护对话历史：
    - system_prompt 只在开始时设置一次
    - 工具结果以 Message 形式添加到历史中
    - 避免重复传递大量示例，降低 token 消耗

    示例：
        from src.app.agents import ReactAgent
        from src.app.agents.tools import ToolRegistry

        tools = {
            "retrieve": ToolRegistry.get_instance("retrieve"),
            "generate": ToolRegistry.get_instance("generate"),
            "classify": ToolRegistry.get_instance("classify"),
            "validate": ToolRegistry.get_instance("validate"),
        }
        agent = ReactAgent(tools, max_steps=5)
        result = await agent.process("2024年泸州雨露计划补贴标准")
        print(result["answer"])
        # 或查看完整消息历史
        for msg in result["messages"]:
            print(f"{msg['role']}: {msg['content'][:50]}...")
    """

    def __init__(
        self,
        tools: Dict[str, BaseTool],
        max_steps: int = 5,
        verbose: bool = False
    ):
        """
        初始化 ReactAgent

        Args:
            tools: 工具字典，键为工具名称，值为工具实例
            max_steps: 最大推理步数（默认 5）
            verbose: 是否开启详细日志（默认 False）
        """
        self.tools = tools
        self.max_steps = max_steps
        self.verbose = verbose
        self._initialized = False
        self.messages: List[Message] = []  # 统一维护消息历史
        self.system_prompt = self._build_system_prompt()  # 一次性构建

        logger.info(
            f"✅ ReactAgent 初始化完成 (max_steps={max_steps}, "
            f"tools={list(tools.keys())})"
        )

    def _build_system_prompt(self) -> str:
        """构建系统提示（只构建一次，降低 token 消耗）"""
        tools_info = self._build_tools_description()

        return f"""# 角色定义
你是一位专业的政务问答专家，善于分析用户问题、检索相关政策、生成专业回答。
你的职责是：提供准确、权威、符合政策的政务咨询服务。

# 任务描述
你需要通过思考(Thought)、行动(Action)、观察(Observation)的循环来回答用户问题。
每一步都要基于已有的信息进行逻辑推理，最终给出准确、完整的回答。

# 可用工具（仔细阅读每个工具的用途和参数）
{tools_info}

# 输出格式（严格遵循 format，不要包含额外文本）
Thought: [基于当前状态的分析和推理，说明你需要做什么]
Action: [工具名称或 'Final Answer']
Action Input: {{key: "value"}}

# 关键原则
1. THOUGHT 要详细：说明为什么选择这个工具，期望获得什么信息
2. ACTION 必须是可用工具之一：{', '.join(self.tools.keys())}
3. ACTION INPUT 必须是 JSON 格式
4. 完整流程：通常需要 retrieve → generate → Final Answer
   - 第1步：总是使用 retrieve 检索相关信息
   - 第2步：基于检索结果使用 generate 生成回答
   - 第3步：使用 Final Answer 提交最终答案
5. 禁止连续检索：retrieve 后必须跟着 generate
6. 使用 Final Answer 结束：当有完整答案时才使用 Final Answer
7. 如果检索结果为空或不相关，尝试使用更通用的检索词后立即生成
8. 如果多次检索无效，在 generate 中说明信息来源限制
"""

    def _build_tools_description(self) -> str:
        """构建工具描述"""
        tools_info_parts = []
        for tool_name, tool in self.tools.items():
            schema = tool.get_schema()
            tool_desc = f"## {tool_name}\n描述：{tool.description}"
            if "parameters" in schema:
                tool_desc += "\n参数说明："
                for param, desc in schema["parameters"].items():
                    tool_desc += f"\n- {param}: {desc}"

            # 添加使用场景说明
            tool_desc += "\n使用场景："
            if tool_name == "retrieve":
                tool_desc += "\n  - 当需要查找相关政策、法规、案例时"
                tool_desc += "\n  - 当需要获取具体的时间、标准、流程等信息时"
            elif tool_name == "generate":
                tool_desc += "\n  - 当已有足够信息可以回答问题时"
                tool_desc += "\n  - 当需要整理、归纳、总结信息时"
            elif tool_name == "classify":
                tool_desc += "\n  - 当需要判断用户提问类型时"
            elif tool_name == "validate":
                tool_desc += "\n  - 当需要检查回答质量时"

            tools_info_parts.append(tool_desc)

        return "\n\n".join(tools_info_parts)

    async def process(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        执行 ReAct 推理循环

        Args:
            query: 用户查询
            **kwargs: 其他参数

        Returns:
            {
                "answer": str,              # 最终答案
                "messages": List[Dict],     # 完整消息历史
                "steps_history": List[Dict],  # 推理步骤历史
                "steps_count": int,           # 推理步数
                "sources": List[Dict],        # 检索来源
                "retrieval_time": float       # 检索耗时
            }
        """
        logger.info(f"🚀 [ReactAgent] 开始处理查询: {query[:50]}...")

        # 初始化消息列表（只在开始时设置一次）
        self.messages = [
            Message(role=SYSTEM, content=self.system_prompt),
            Message(role=USER, content=query)
        ]

        # 循环执行 Thought-Action-Observation
        final_answer = ""
        sources = []
        retrieval_time = 0.0

        for step_count in range(self.max_steps):
            step_number = step_count + 1

            # ========== Thought: LLM 分析当前状态 ==========
            logger.debug(f"💭 [Step {step_number}] 生成思考...")

            # LLM 调用（维护 Message 数组）
            self.messages = await self._llm_call(self.messages)

            # 解析 LLM 响应
            last_msg = self.messages[-1]
            thought, action, action_input = self._parse_thought_action(last_msg)

            logger.debug(f"  → 思考: {thought[:100]}...")
            logger.debug(f"  → 动作: {action} | 输入: {action_input}")

            # ========== Action: 执行工具 ==========
            logger.debug(f"⚙️ [Step {step_number}] 执行动作: {action}")

            observation, execution_time = await self._execute_tool(action, action_input)

            # 记录工具执行日志
            logger.debug(f"  → 执行耗时: {execution_time:.2f}s | 观察: {observation[:50]}...")

            # 添加工具结果到消息历史
            fn_msg = Message(
                role=FUNCTION,
                name=action,
                content=observation,
                extra={"step_number": step_number, "execution_time": execution_time}
            )
            self.messages.append(fn_msg)

            # 记录检索结果
            if action == "retrieve":
                try:
                    result = json.loads(observation)
                    if "results" in result:
                        sources.extend(result["results"])
                    if "metadata" in result and "retrieval_time" in result["metadata"]:
                        retrieval_time = result["metadata"]["retrieval_time"]
                except json.JSONDecodeError:
                    pass

            if self.verbose:
                logger.info(f"📊 Step {step_number}: {action} → {observation[:50]}...")

            # ========== 验证与重试逻辑 ==========
            if action == "generate" and "validate" in self.tools:
                validation_passed = await self._validate_and_retry_if_needed(
                    query=query,
                    answer=observation,
                    context=action_input.get("context", "")
                )
                if not validation_passed:
                    logger.warning(f"  ⚠️ 验证失败，尝试继续推理...")

            # ========== 判断是否结束 ==========================
            if action == "Final Answer":
                final_answer = action_input.get("answer", observation)
                logger.info(f"✅ [ReactAgent] 生成最终答案 (步数: {step_number})")
                break

        # 如果达到最大步数仍未生成最终答案，强制生成
        if not final_answer:
            logger.warning(f"⚠️  达到最大步数 ({self.max_steps})，强制生成答案")
            final_answer = await self._generate_final_answer()

        return self._build_result(final_answer, sources, retrieval_time)

    async def _llm_call(self, messages: List[Message]) -> List[Message]:
        """
        与 LLM 交互，返回更新后的消息列表

        Args:
            messages: 当前消息列表

        Returns:
            更新后的消息列表（包含 LLM 响应）
        """
        try:
            # 转换 Message 为 dict 列表（dashscope API 要求）
            messages_dict = [msg.model_dump() for msg in messages]

            # 调用 LLM（使用优化模型）
            optimizer_llm = get_optimizer_llm_service()
            response = Generation.call(
                model=optimizer_llm.get_model_name(),
                messages=messages_dict,
                temperature=optimizer_llm.get_config().temperature,
                max_tokens=5000,
                top_p=optimizer_llm.get_config().top_p,
                result_format='message'  # 返回 Message 格式
            )

            if response.status_code == 200:
                # 解析 LLM 响应
                choice = response.output.choices[0]
                response_msg = choice.get('message', {})

                # 构建 Message 对象（兼容 Dashscope 格式）
                msg = Message(
                    role=response_msg.get('role', 'assistant'),
                    content=response_msg.get('content', ''),
                    function_call=response_msg.get('function_call')
                )
                messages.append(msg)

            return messages

        except Exception as e:
            logger.error(f"❌ LLM 调用失败: {e}")
            logger.error(traceback.format_exc())
            raise

    def _parse_thought_action(
        self,
        message: Message
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        从 Message 中解析 Thought、Action、Action Input

        Args:
            message: LLM 响应的 Message 对象

        Returns:
            (thought, action, action_input)
        """
        content = message.content or ""

        # 尝试解析 JSON 格式的 function_call
        if message.function_call:
            func_call = message.function_call
            func_name = func_call.name
            func_args = func_call.arguments
            try:
                action_input = json.loads(func_args)
            except json.JSONDecodeError:
                action_input = {"query": func_args}

            # 从 content 中提取 thought（如果有）
            thought = ""
            if "\nAction:" in content:
                thought = content.split("\nAction:")[0].strip()
            if not thought:
                thought = f"使用 {func_name} 工具来处理查询"

            return thought, func_name, action_input

        # 解析文本格式的 Thought/Action/Action Input
        thought = ""
        action = "Final Answer"
        action_input = {}

        lines = content.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 解析 Thought
            if line.startswith("Thought:"):
                thought = line[8:].strip()
            elif line.startswith("Thought "):
                # 处理 "Thought 1:" 等格式
                idx = line.find(":")
                if idx > 0:
                    thought = line[idx+1:].strip()
            # 解析 Action
            elif line.startswith("Action:"):
                action = line[7:].strip()
            elif line.startswith("Action "):
                idx = line.find(":")
                if idx > 0:
                    action = line[idx+1:].strip()
            # 解析 Action Input
            elif line.startswith("Action Input:"):
                try:
                    input_str = line[13:].strip()
                    action_input = json.loads(input_str)
                except json.JSONDecodeError:
                    # 尝试提取 JSON 对象
                    json_match = self._extract_json(input_str)
                    if json_match:
                        try:
                            action_input = json.loads(json_match)
                        except json.JSONDecodeError:
                            action_input = {}
                    else:
                        action_input = {}
            elif line.startswith("Action Input "):
                idx = line.find(":")
                if idx > 0:
                    try:
                        input_str = line[idx+1:].strip()
                        json_match = self._extract_json(input_str)
                        if json_match:
                            action_input = json.loads(json_match)
                    except json.JSONDecodeError:
                        action_input = {}

        # 验证工具名称
        if action != "Final Answer" and action not in self.tools:
            logger.warning(f"⚠️  未知工具: {action}，使用默认工具 'retrieve'")
            action = "retrieve"
            action_input = {"query": content}

        return thought, action, action_input

    def _extract_json(self, text: str) -> Optional[str]:
        """从文本中提取 JSON 对象"""
        import re
        # 尝试匹配 JSON 对象
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
        if json_match:
            return json_match.group()
        return None

    def _build_result(
        self,
        final_answer: str,
        sources: List[Dict],
        retrieval_time: float
    ) -> Dict[str, Any]:
        """
        从 Message 数组构建结果

        Args:
            final_answer: 最终答案
            sources: 检索来源列表
            retrieval_time: 检索耗时

        Returns:
            结果字典
        """
        # 从 messages 中构建 steps_history
        steps_history = []
        current_step = None

        for i, msg in enumerate(self.messages):
            if msg.role == "assistant" and msg.content:
                # 解析 assistant 消息中的 Thought/Action
                thought, action, action_input = self._parse_thought_action(msg)

                if action != "Final Answer":
                    current_step = {
                        "step_number": len(steps_history) + 1,
                        "thought": thought,
                        "action": action,
                        "action_input": action_input,
                        "timestamp": datetime.now().isoformat()
                    }
                    steps_history.append(current_step)
                else:
                    # Final Answer 不创建新步骤
                    if current_step:
                        current_step["final_answer"] = final_answer
            elif msg.role == "function":
                # 处理工具结果
                if current_step and msg.name == "retrieve":
                    try:
                        result = json.loads(msg.content)
                        if "results" in result:
                            current_step["sources"] = result["results"]
                    except json.JSONDecodeError:
                        pass

        return {
            "answer": final_answer,
            "messages": [msg.model_dump() for msg in self.messages],
            "steps_history": steps_history,
            "steps_count": len(steps_history),
            "sources": sources,
            "retrieval_time": retrieval_time
        }

    def _validate_tool_input(
        self,
        action: str,
        action_input: Dict[str, Any]
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
        requirements = TOOL_PARAMETER_REQUIREMENTS.get(action, {})
        required_params = requirements.get("required", [])

        for param in required_params:
            if param not in action_input:
                return False, f"缺少必要参数: {param}"

        # 应用默认值
        defaults = requirements.get("defaults", {})
        for param, default_value in defaults.items():
            if param not in action_input:
                action_input[param] = default_value

        return True, ""

    async def _execute_tool(
        self,
        action: str,
        action_input: Dict[str, Any]
    ) -> Tuple[str, float]:
        """
        执行工具并格式化结果

        Args:
            action: 动作名称
            action_input: 动作参数

        Returns:
            (observation, execution_time)
        """
        from datetime import datetime

        start_time = datetime.now()

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
        max_retries = 1  # TOOL_MAX_RETRIES
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

    def _format_observation(self, result: Dict[str, Any]) -> str:
        """
        格式化工具执行结果为自然语言描述

        Args:
            result: 工具执行结果

        Returns:
            格式化后的观察文本
        """
        if "answer" in result:
            return result["answer"]

        if "results" in result:
            results = result["results"]
            if isinstance(results, list) and len(results) > 0:
                lines = ["检索到以下相关案例："]
                for idx, item in enumerate(results[:5], 1):
                    title = item.get("title", "无标题")
                    dept = item.get("department", "未知部门")
                    lines.append(f"{idx}. {title} ({dept})")
                return "\n".join(lines)

        if "type" in result:
            type_ = result.get("type", "未知")
            confidence = result.get("confidence", 0.0)
            return f"问政类型: {type_}, 置信度: {confidence:.2f}"

        if "overall_score" in result:
            score = result.get("overall_score", 0.0)
            passed = result.get("passed", False)
            return f"质量评分: {score:.2f}, 通过: {passed}"

        # 默认格式化
        return json.dumps(result, ensure_ascii=False)

    async def _generate_final_answer(self) -> str:
        """
        使用主模型生成最终答案（基于 Message 历史）

        Returns:
            最终答案文本
        """
        # 从 messages 中提取检索结果
        context_parts = []
        for msg in self.messages:
            if msg.role == "function" and msg.name == "retrieve":
                try:
                    result = json.loads(msg.content)
                    if "results" in result:
                        for item in result["results"][:5]:
                            title = item.get("title", item.get("name", "无标题"))
                            content = item.get("content", item.get("text", ""))
                            department = item.get("department", item.get("unit", ""))
                            if content:
                                context_parts.append(f"【{department}】{title}\n{content}")
                except json.JSONDecodeError:
                    pass

        context = "\n\n".join(context_parts)

        # 如果没有上下文，使用默认回答
        if not context.strip():
            logger.warning("⚠️  没有检索到相关上下文，将基于常识回答")

            # 从 messages 中提取用户查询
            user_query = ""
            for msg in self.messages:
                if msg.role == "user":
                    user_query = msg.content
                    break

            prompt = f"""用户查询：{user_query}

由于没有检索到相关资料，以下回答基于一般常识，请谨慎参考：
"""
        else:
            # 从 messages 中提取用户查询
            user_query = ""
            for msg in self.messages:
                if msg.role == "user":
                    user_query = msg.content
                    break

            prompt = f"""基于以下检索结果回答问题：

检索结果：
{context}

问题：{user_query}

请给出准确、完整的回答。如果检索结果不相关，请说明原因。
"""

        # 使用主模型生成答案
        try:
            heavy_llm = get_heavy_llm_service()
            logger.debug(f"📝 [Final Answer] 使用上下文长度: {len(context)}")

            response = Generation.call(
                model=heavy_llm.get_model_name(),
                prompt=prompt,
                temperature=heavy_llm.get_config().temperature,
                max_tokens=heavy_llm.get_config().max_tokens,
                top_p=heavy_llm.get_config().top_p,
                result_format='text'
            )

            if response.status_code == 200:
                return response.output.text
            else:
                return "抱歉，生成最终答案时出现错误。"

        except Exception as e:
            logger.error(f"❌ 生成最终答案失败: {e}")
            logger.error(traceback.format_exc())
            return f"抱歉，生成最终答案时出现错误：{str(e)}"

    async def _validate_and_retry_if_needed(
        self,
        query: str,
        answer: str,
        context: str = ""
    ) -> bool:
        """
        验证回答质量，如果质量低则尝试重新生成

        Args:
            query: 用户查询
            answer: 生成的回答
            context: 上下文信息

        Returns:
            是否验证通过
        """
        if "validate" not in self.tools:
            return True  # 没有验证工具，直接返回

        try:
            validate_tool = self.tools["validate"]
            validation = await validate_tool.execute(
                answer=answer,
                query=query,
                context=context
            )

            overall_score = validation.get("overall_score", 0.0)
            passed = validation.get("passed", False)
            feedback = validation.get("feedback", "")

            logger.debug(f"🔍 验证结果: {overall_score:.2f} (通过: {passed})")

            if not passed or overall_score < 0.7:
                logger.warning(f"  ⚠️ 验证未通过，反馈: {feedback[:50]}...")

                # 记录验证结果到最近的 function message
                for msg in reversed(self.messages):
                    if msg.role == "function" and msg.name == "generate":
                        if msg.extra is None:
                            msg.extra = {}
                        msg.extra["validation_result"] = validation
                        break

                return False

            return True

        except Exception as e:
            logger.error(f"❌ 验证过程出错: {e}")
            return True  # 验证失败不影响流程继续

    async def initialize(self) -> None:
        """初始化所有工具（预热）"""
        logger.info("⏳ [ReactAgent] 初始化工具...")
        for tool_name, tool in self.tools.items():
            if hasattr(tool, 'initialize') and callable(tool.initialize):
                try:
                    await tool.initialize()
                    logger.info(f"  ✅ {tool_name} 已预热")
                except Exception as e:
                    logger.warning(f"  ⚠️ {tool_name} 预热失败: {e}")
        self._initialized = True
        logger.info("✅ [ReactAgent] 初始化完成")

    def get_status(self) -> Dict[str, Any]:
        """获取 Agent 状态"""
        return {
            "initialized": self._initialized,
            "max_steps": self.max_steps,
            "tools": list(self.tools.keys()),
            "verbose": self.verbose,
            "message_count": len(self.messages),
            "messages_summary": [
                {"role": msg.role, "content_length": len(str(msg.content))}
                for msg in self.messages
            ]
        }

    # ==================== 与 qwen_agent 兼容的工具调用检测 ====================

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
            text = "".join(item.value for item in text if hasattr(item, 'value'))

        # 解析 Thought/Action/Action Input
        special_func_token = '\nAction:'
        special_args_token = '\nAction Input:'
        func_name, func_args = None, None

        i = text.rfind(special_func_token)
        j = text.rfind(special_args_token)

        if 0 <= i < j:
            func_name = text[i + len(special_func_token):j].strip()
            func_args = text[j + len(special_args_token):].strip()
            try:
                func_args = json.loads(func_args)
            except json.JSONDecodeError:
                func_args = {"query": func_args}
            return True, func_name, func_args

        return False, "", {}

    def _format_tool_result(self, tool_name: str, result: Any) -> Message:
        """
        格式化工具结果为 Message

        Args:
            tool_name: 工具名称
            result: 工具执行结果

        Returns:
            Message 对象（role="function"）
        """
        return Message(
            role="function",
            name=tool_name,
            content=self._format_observation(result),
            extra={"function_id": "1"}
        )

    def _build_prompt_from_messages(self, messages: List[Message]) -> str:
        """
        从 Message 列表构建 Prompt 字符串

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
