"""
ReAct Agent 核心实现
Thought-Action-Observation 循环引擎

ReAct范式：
1. Thought: LLM 分析当前状态并生成思考
2. Action: 选择并执行工具
3. Observation: 获取工具执行结果
4. 循环直到生成最终答案或达到最大步数

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
"""

import json
import traceback
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Protocol
from pydantic import BaseModel
from src.app.infra.llm.multi_model_service import (
    get_optimizer_llm_service,
    get_heavy_llm_service
)
from src.app.infra.utils.logger import get_logger
from dashscope import Generation

logger = get_logger(__name__)


class BaseTool(Protocol):
    """工具协议 - 定义 ReAct 工具接口"""

    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...

    async def execute(self, **kwargs) -> Dict[str, Any]: ...


class ReactStep(BaseModel):
    """
    ReAct 推理步骤记录

    记录每一步的思考、行动、观察结果
    """

    step_number: int              # 步骤编号
    thought: str                  # 思考内容
    action: str                   # 动作名称
    action_input: Dict[str, Any]  # 动作参数
    observation: str              # 观察结果
    timestamp: datetime           # 时间戳

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def __repr__(self):
        return f"Step {self.step_number}: {self.action}"


class ReactAgent:
    """
    ReAct Agent 核心类

    实现 Thought-Action-Observation 循环：
    1. Thought: LLM 分析当前状态并生成思考
    2. Action: 选择并执行工具
    3. Observation: 获取工具执行结果
    4. 循环直到生成最终答案或达到最大步数

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

        logger.info(
            f"✅ ReactAgent 初始化完成 (max_steps={max_steps}, "
            f"tools={list(tools.keys())})"
        )

    async def process(
        self,
        query: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        执行 ReAct 推理循环

        Args:
            query: 用户查询
            **kwargs: 其他参数

        Returns:
            {
                "answer": str,              # 最终答案
                "steps_history": List[Dict],  # 推理步骤历史
                "steps_count": int,           # 推理步数
                "sources": List[Dict],        # 检索来源
                "retrieval_time": float       # 检索耗时
            }
        """
        logger.info(f"🚀 [ReactAgent] 开始处理查询: {query[:50]}...")

        # 初始化步骤历史
        steps_history: List[ReactStep] = []

        # 循环执行 Thought-Action-Observation
        final_answer = ""
        sources = []
        retrieval_time = 0.0

        for step_count in range(self.max_steps):
            step_number = step_count + 1

            # ========== Thought: LLM 分析当前状态 ==========
            logger.debug(f"💭 [Step {step_number}] 生成思考...")

            thought, action, action_input = await self._generate_thought_and_action(
                query=query,
                steps_history=steps_history
            )

            logger.debug(f"  → 思考: {thought[:100]}...")
            logger.debug(f"  → 动作: {action} | 输入: {action_input}")

            # ========== Action: 执行工具 ==========
            logger.debug(f"⚙️ [Step {step_number}] 执行动作: {action}")

            observation = await self._execute_tool(action, action_input)

            # 记录检索结果
            if action == "retrieve":
                if "results" in action_input and isinstance(action_input["results"], list):
                    sources.extend(action_input["results"])
                if "metadata" in action_input and "retrieval_time" in action_input["metadata"]:
                    retrieval_time = action_input["metadata"]["retrieval_time"]

            # ========== 记录步骤 ==========
            step = ReactStep(
                step_number=step_number,
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation,
                timestamp=datetime.now()
            )
            steps_history.append(step)

            if self.verbose:
                logger.info(f"📊 Step {step_number}: {action} → {observation[:50]}...")

            # ========== 判断是否结束 ==========
            if action == "Final Answer":
                final_answer = action_input.get("answer", observation)
                logger.info(f"✅ [ReactAgent] 生成最终答案 (步数: {step_number})")
                break

        # 如果达到最大步数仍未生成最终答案，强制生成
        if not final_answer:
            logger.warning(f"⚠️  达到最大步数 ({self.max_steps})，强制生成答案")
            final_answer = await self._generate_final_answer(query, steps_history)

        # 序列化步骤历史
        serialized_steps = [step.model_dump() for step in steps_history]

        return {
            "answer": final_answer,
            "steps_history": serialized_steps,
            "steps_count": len(steps_history),
            "sources": sources,
            "retrieval_time": retrieval_time
        }

    async def _generate_thought_and_action(
        self,
        query: str,
        steps_history: List[ReactStep]
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        生成思考和动作

        Args:
            query: 用户查询
            steps_history: 步骤历史

        Returns:
            (thought, action, action_input)
        """
        # 构建 ReAct 提示
        prompt = self._build_react_prompt(query, steps_history)

        try:
            # 使用优化模型生成思考和动作
            optimizer_llm = get_optimizer_llm_service()
            response = Generation.call(
                model=optimizer_llm.get_model_name(),
                prompt=prompt,
                temperature=optimizer_llm.get_config().temperature,
                max_tokens=500,
                top_p=optimizer_llm.get_config().top_p,
                result_format='text'
            )

            if response.status_code == 200:
                response_text = response.output.text
                thought, action, action_input = self._parse_react_response(response_text)
                return thought, action, action_input
            else:
                raise Exception(f"API调用失败: {response.code} - {response.message}")

        except Exception as e:
            logger.error(f"❌ 生成思考和动作失败: {e}")
            logger.error(traceback.format_exc())
            # 返回默认动作（检索）
            return (
                f"需要检索相关信息来回答问题: {query}",
                "retrieve",
                {"query": query, "top_k": 5}
            )

    def _build_react_prompt(
        self,
        query: str,
        steps: List[ReactStep]
    ) -> str:
        """
        构建 ReAct 提示模板

        Args:
            query: 用户查询
            steps: 步骤历史

        Returns:
            完整的提示文本
        """
        parts = []

        # 任务描述
        parts.append("# 任务描述")
        parts.append("你是一个 ReAct Agent，需要通过思考(Thought)、行动(Action)、观察(Observation)的循环来回答问题。")
        parts.append("")
        parts.append("# 可用工具")
        for tool_name, tool in self.tools.items():
            parts.append(f"- {tool_name}: {tool.description}")
        parts.append("")
        parts.append("# 输出格式（严格按照以下格式）")
        parts.append("Thought: [基于当前状态的思考]")
        parts.append("Action: [工具名称或 'Final Answer']")
        parts.append("Action Input: {\"key\": \"value\"}")
        parts.append("")
        parts.append("# 注意事项")
        parts.append("- Action 必须是可用工具之一，或 'Final Answer'")
        parts.append("- Action Input 必须是有效的 JSON 对象")
        parts.append("- Final Answer 用于生成最终回答")
        parts.append("")

        # 对话历史
        if steps:
            parts.append("# 推理历史")
            for step in steps:
                parts.append(f"Thought {step.step_number}: {step.thought}")
                parts.append(f"Action {step.step_number}: {step.action}")
                parts.append(f"Action Input {step.step_number}: {json.dumps(step.action_input, ensure_ascii=False)}")
                observation_preview = step.observation[:200] if len(step.observation) > 200 else step.observation
                parts.append(f"Observation {step.step_number}: {observation_preview}")
                parts.append("")
        else:
            parts.append("# 推理历史")
            parts.append("无")
            parts.append("")

        # 当前查询
        parts.append("# 当前查询")
        parts.append(query)
        parts.append("")

        # 当前步骤
        parts.append("# 当前步骤")
        parts.append("Thought:")

        return "\n".join(parts)

    def _parse_react_response(
        self,
        response: str
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        解析 LLM 响应为 Thought、Action、Action Input

        Args:
            response: LLM 响应文本

        Returns:
            (thought, action, action_input)
        """
        thought = ""
        action = "Final Answer"
        action_input = {}

        lines = response.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("Thought:") or line.startswith("Thought："):
                thought = line.split(":", 1)[1].strip() if ":" in line else line.split("：", 1)[1].strip()
            elif line.startswith("Action:") or line.startswith("Action："):
                action = line.split(":", 1)[1].strip() if ":" in line else line.split("：", 1)[1].strip()
            elif line.startswith("Action Input:") or line.startswith("Action Input："):
                try:
                    input_str = line.split(":", 1)[1].strip() if ":" in line else line.split("：", 1)[1].strip()
                    action_input = json.loads(input_str)
                except json.JSONDecodeError:
                    # 如果解析失败，使用默认输入
                    action_input = {"query": response}

        # 如果没有提取到思考，使用响应作为思考
        if not thought:
            thought = "分析用户查询并选择合适的工具"

        # 验证工具名称
        if action != "Final Answer" and action not in self.tools:
            logger.warning(f"⚠️  未知工具: {action}，使用默认工具 'retrieve'")
            action = "retrieve"
            action_input = {"query": response}

        return thought, action, action_input

    async def _execute_tool(
        self,
        action: str,
        action_input: Dict[str, Any]
    ) -> str:
        """
        执行工具并格式化结果

        Args:
            action: 动作名称
            action_input: 动作参数

        Returns:
            观察结果（格式化为文本）
        """
        if action == "Final Answer":
            answer = action_input.get("answer", "")
            return answer

        if action not in self.tools:
            return f"错误：未知工具 {action}"

        try:
            # 执行工具
            tool = self.tools[action]

            # 特殊处理：GenerationTool 需要 prompt 参数
            if action == "generate":
                if "prompt" not in action_input:
                    if "query" in action_input:
                        action_input["prompt"] = action_input["query"]
                    elif "answer" in action_input:
                        action_input["prompt"] = action_input["answer"]
                    else:
                        action_input["prompt"] = ""

            result = await tool.execute(**action_input)

            # 格式化观察结果
            observation = self._format_observation(result)
            return observation

        except Exception as e:
            logger.error(f"❌ 工具执行失败: {e}")
            logger.error(traceback.format_exc())
            return f"错误：{str(e)}"

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

    async def _generate_final_answer(
        self,
        query: str,
        steps_history: List[ReactStep]
    ) -> str:
        """
        使用主模型生成最终答案

        Args:
            query: 用户查询
            steps_history: 步骤历史

        Returns:
            最终答案文本
        """
        # 构建上下文
        context_parts = []
        for step in steps_history:
            if step.action == "retrieve" and "results" in step.action_input:
                results = step.action_input["results"]
                for item in results[:3]:
                    title = item.get("title", "")
                    content = item.get("content", "")
                    context_parts.append(f"{title}: {content}")

        context = "\n\n".join(context_parts)

        # 使用主模型生成答案
        try:
            heavy_llm = get_heavy_llm_service()
            prompt = f"""
基于以下上下文回答问题：

上下文：
{context}

问题：{query}

请给出准确、完整的回答。
"""

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
            return "抱歉，生成最终答案时出现错误。"

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
            "verbose": self.verbose
        }
