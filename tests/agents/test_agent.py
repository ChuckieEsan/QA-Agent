"""
LangGraph Agent 测试

用法：
    cd /home/liuchenyu/QA-Agent
    python -m tests.agents.test_agent
"""

import asyncio
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.app.agents import run_agent, ainvoke
from src.app.agents.state import ProcessStatus


def test_agent():
    """测试 Agent 执行"""
    print("=" * 60)
    print("测试 LangGraph MultiAgent 工作流")
    print("=" * 60)

    query = "泸州雨露计划补贴标准是多少？"
    print(f"\n📝 查询：{query}")

    try:
        result = asyncio.run(run_agent(query))
        print("\n✅ 调用成功!")

        status = result.get("status")
        print(f"\n📋 最终回复：{result.get('generated_response', 'N/A')[:200]}...")
        print(f"\n📊 状态信息:")
        print(f"   - 状态：{status}")
        print(f"   - 诉求类型：{result.get('classification', {}).get('request_type', 'N/A')}")
        print(f"   - 紧急程度：{result.get('classification', {}).get('request_urgency', 'N/A')}")
        print(f"   - 置信度：{result.get('confidence_score', 0.0):.2f}")
        print(f"   - 检索结果数：{len(result.get('retrieved_knowledge', []))}")
        return True
    except Exception as e:
        print(f"\n❌ 调用失败：{e}")
        import traceback
        traceback.print_exc()
        return False


async def test_async_agent():
    """测试异步 Agent 执行"""
    print("=" * 60)
    print("测试异步调用 LangGraph MultiAgent 工作流")
    print("=" * 60)

    query = "泸州雨露计划补贴标准是多少？"
    print(f"\n📝 查询：{query}")

    try:
        result = await ainvoke(query)
        print("\n✅ 调用成功!")
        print(f"\n📋 最终回复：{result.get('generated_response', 'N/A')[:200]}...")
        return True
    except Exception as e:
        print(f"\n❌ 调用失败：{e}")
        return False


if __name__ == "__main__":
    # 设置日志级别
    import logging
    logging.getLogger("src").setLevel(logging.INFO)

    # 测试
    test_agent()

    # 测试异步
    asyncio.run(test_async_agent())

    print("\n" + "=" * 60)
    print("所有测试完成")
    print("=" * 60)