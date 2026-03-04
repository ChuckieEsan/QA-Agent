#!/usr/bin/env python
"""
ReAct 框架演示脚本
展示 ReAct Agent 的 Thought-Action-Observation 循环
"""

import asyncio
import sys
from typing import Dict, Any

from src.app.infra.utils.logger import get_logger
logger = get_logger(__name__)

sys.path.append(".")

from src.app.agents import ReactAgent
from src.app.agents.tools import ToolRegistry

tools = ToolRegistry.list_all()

async def demo_react_agent_basic():
    """演示基础 ReAct Agent（手动创建工具）"""
    logger.info("=" * 80)
    logger.info("演示 1: 基础 ReAct Agent (手动创建工具)")
    logger.info("=" * 80)

    # 创建 ReAct Agent（最大步数 3）
    agent = ReactAgent(tools)

    # 执行推理
    query = "2024年泸州雨露计划补贴标准"
    logger.info(f"用户查询: {query}")

    result = await agent.process(query)
    logger.debug(result)


async def demo_react_agent_simple():
    """演示简易 ReAct Agent（自动创建工具）"""
    logger.info("=" * 80)
    logger.info("演示 2: 简易 ReAct Agent (自动创建工具)")
    logger.info("=" * 80)

    # 创建工具集
    # 创建 ReAct Agent
    agent = ReactAgent(tools, max_steps=5)

    # 执行推理
    query = "泸州小微企业有哪些税收优惠政策？"
    logger.info(f"用户查询: {query}")

    result = await agent.process(query)

    # 显示完整输出
    logger.info("推理统计:")
    logger.info("-" * 80)
    logger.info(f"推理步数: {result['steps_count']}")
    logger.info(f"检索耗时: {result['retrieval_time']:.2f}s")
    logger.info(f"检索来源: {len(result['sources'])} 个")
    print("-" * 80)

    # 显示推理历史
    print(f"\n🔍 推理步骤 ({len(result['steps_history'])} 步):")
    print("-" * 80)
    for step in result["steps_history"]:
        print(f"\n【步骤 {step['step_number']}】")
        print(f"💭 思考: {step['thought']}")
        print(f"⚙️  动作: {step['action']}")
        print(f"👀 观察: {step['observation'][:100]}...")
    print("-" * 80)

    # 显示最终答案
    print(f"\n🎯 最终答案:")
    print("-" * 80)
    print(result["answer"])
    print("-" * 80)


async def demo_multi_complexity_queries():
    """演示不同复杂度的查询"""
    print("\n" + "=" * 80)
    print("📋 演示 3: 不同复杂度的查询对比")
    print("=" * 80)

    # 创建工具集
    agent = ReactAgent(tools, max_steps=5)

    test_cases = [
        {
            "query": "如何办理身份证？",
            "description": "【简单查询】常识性问题"
        },
        {
            "query": "2024年泸州雨露计划补贴标准",
            "description": "【中等复杂度】需要检索政策"
        },
        {
            "query": "泸州小微企业税收优惠和社保补贴政策",
            "description": "【高复杂度】跨部门政策"
        }
    ]

    for case in test_cases:
        print(f"\n{case['description']}")
        print(f"❓ 问题: {case['query']}")
        print("-" * 80)

        result = await agent.process(case["query"])

        print(f"📊 推理步数: {result['steps_count']}")
        print(f"📊 检索来源: {len(result['sources'])} 个")
        print(f"✅ 答案: {result['answer'][:150]}...")
        print("-" * 80)


async def demo_thought_action_observation_loop():
    """详细展示 Thought-Action-Observation 循环"""
    print("\n" + "=" * 80)
    print("📋 演示 4: 详细 Thought-Action-Observation 循环")
    print("=" * 80)

    agent = ReactAgent(tools, max_steps=5)

    query = "泸州创业扶持政策有哪些？"
    print(f"\n❓ 用户查询: {query}\n")

    result = await agent.process(query)

    print("🔄 完整循环过程:")
    print("=" * 80)

    for step in result["steps_history"]:
        step_num = step['step_number']
        print(f"\n{'='*80}")
        print(f"🔴 步骤 {step_num}")
        print(f"{'='*80}")

        # Thought
        print(f"\n💭 [THOUGHT]")
        print(f"   {step['thought']}")

        # Action
        print(f"\n⚙️  [ACTION]")
        print(f"   工具: {step['action']}")
        print(f"   输入: {step['action_input']}")

        # Observation
        print(f"\n👀 [OBSERVATION]")
        print(f"   {step['observation']}")

    print(f"\n{'='*80}")
    print(f"✅ 最终答案 ({result['steps_count']} 步)")
    print(f"{'='*80}")
    print(result["answer"])


async def demo_tool_extension():
    """演示工具扩展 - 添加自定义工具"""
    print("\n" + "=" * 80)
    print("📋 演示 5: 工具扩展 - 自定义工具")
    print("=" * 80)

    from src.app.agents.tools.base_tool import BaseTool

    # 定义自定义工具
    class EchoTool(BaseTool):
        name = "echo"
        description = "回显用户输入"

        async def execute(self, text: str = "") -> Dict[str, Any]:
            return {"result": f"Echo: {text}"}

    # 创建工具集并添加自定义工具
    tools["echo"] = EchoTool()

    agent = ReactAgent(tools, max_steps=3)

    query = "测试回显功能：你好"
    print(f"\n❓ 用户查询: {query}\n")

    result = await agent.process(query)

    print("🔍 推理步骤:")
    for step in result["steps_history"]:
        print(f"  Step {step['step_number']}: {step['action']}")

    print(f"\n✅ 最终答案: {result['answer'][:100]}...")


async def main():
    """主函数"""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 25 + "ReAct 框架演示" + " " * 40 + "║")
    print("║" + " " * 15 + "Thought-Action-Observation 循环引擎" + " " * 28 + "║")
    print("╚" + "═" * 78 + "╝")

    print("\n" + "📚 简介:")
    print("  ReAct 框架通过 Thought-Action-Observation 循环实现动态推理:")
    print("  1. Thought: LLM 分析当前状态并生成思考")
    print("  2. Action: 选择并执行合适的工具")
    print("  3. Observation: 获取工具执行结果")
    print("  4. 循环直到生成最终答案")

    # 运行演示
    await demo_react_agent_basic()
    # await demo_react_agent_simple()
    # await demo_multi_complexity_queries()
    # await demo_thought_action_observation_loop()
    # await demo_tool_extension()

    print("\n" + "=" * 80)
    print("✅ 所有演示完成！")
    print("=" * 80)
    print("\n📌 关键特性:")
    print("  ✅ 动态推理循环 (Thought-Action-Observation)")
    print("  ✅ 工具可扩展 (Retrieval/Generation/Classification/Validation)")
    print("  ✅ 完整推理追踪 (每一步的思考、行动、观察)")
    print("  ✅ 多模型优化 (Optimizer + Heavy)")
    print("  ✅ 工具注册表 (ToolRegistry 支持自定义工具)")
    print("  ✅ 安全限制 (最大步数防止无限循环)")


if __name__ == "__main__":
    asyncio.run(main())
