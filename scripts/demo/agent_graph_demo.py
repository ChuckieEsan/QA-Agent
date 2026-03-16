"""
LangGraph Agent 测试 Demo

测试不同的场景：
1. 正常咨询流程
2. 多轮对话
3. 置信度触发兜底
4. 投诉举报类型触发兜底

用法：
    python scripts/demo/agent_graph_demo.py
"""

import asyncio
from src.app.agents import ainvoke, gov_agent_app
from src.app.infra.utils.logger import get_logger

logger = get_logger(__name__)


async def test_normal_consult():
    """测试 1: 正常咨询流程"""
    print("\n" + "=" * 60)
    print("测试 1: 正常咨询流程")
    print("=" * 60)

    query = "雨露计划什么时候发放？"
    print(f"\n📝 查询: {query}")

    result = await ainvoke(query, session_id="test-normal")

    print(f"\n📋 结果:")
    print(f"   - 分类: {result.get('classification', {}).get('request_type', 'N/A')}")
    print(f"   - 置信度: {result.get('confidence_score', 0):.2f}")
    print(f"   - 回复: {result.get('final_reply', 'N/A')[:150]}...")
    print(f"   - 工单ID: {result.get('work_order_id')}")

    return result


async def test_multi_turn():
    """测试 2: 多轮对话"""
    print("\n" + "=" * 60)
    print("测试 2: 多轮对话")
    print("=" * 60)

    session_id = "test-multi-turn"

    # 第一轮
    query1 = "我想咨询社保问题"
    print(f"\n📝 第1轮: {query1}")
    result1 = await ainvoke(query1, session_id=session_id)
    print(f"   回复: {result1.get('final_reply', '')[:80]}...")

    # 第二轮
    query2 = "那灵活就业人员怎么缴纳？"
    print(f"\n📝 第2轮: {query2}")
    result2 = await ainvoke(query2, session_id=session_id)
    print(f"   回复: {result2.get('final_reply', '')[:80]}...")

    # 第三轮
    query3 = "需要提供什么材料？"
    print(f"\n📝 第3轮: {query3}")
    result3 = await ainvoke(query3, session_id=session_id)
    print(f"   回复: {result3.get('final_reply', '')[:80]}...")

    # 查看消息历史
    messages = result3.get("messages", [])
    print(f"\n📊 消息历史: {len(messages)} 条")

    return result3


async def test_complaint():
    """测试 3: 投诉举报类型触发兜底"""
    print("\n" + "=" * 60)
    print("测试 3: 投诉举报类型")
    print("=" * 60)

    query = "我要投诉某部门办事效率太低了！"
    print(f"\n📝 查询: {query}")

    result = await ainvoke(query, session_id="test-complaint")

    print(f"\n📋 结果:")
    print(f"   - 分类: {result.get('classification', {}).get('request_type', 'N/A')}")
    print(f"   - 置信度: {result.get('confidence_score', 0):.2f}")
    print(f"   - 回复: {result.get('final_reply', '')[:100]}...")
    print(f"   - 工单ID: {result.get('work_order_id')}")

    return result


async def test_low_confidence():
    """测试 4: 低置信度触发兜底（模拟）"""
    print("\n" + "=" * 60)
    print("测试 4: 低置信度场景")
    print("=" * 60)

    # 这个查询可能会得到较低的置信度
    query = "量子纠缠原理是什么？"
    print(f"\n📝 查询: {query}（政务无关问题）")

    result = await ainvoke(query, session_id="test-low-conf")

    print(f"\n📋 结果:")
    print(f"   - 分类: {result.get('classification', {}).get('request_type', 'N/A')}")
    print(f"   - 置信度: {result.get('confidence_score', 0):.2f}")
    print(f"   - 回复: {result.get('final_reply', '')[:100]}...")
    print(f"   - 工单ID: {result.get('work_order_id')}")

    return result


async def test_escalation():
    """测试 5: 情绪逃逸（多轮后触发兜底）"""
    print("\n" + "=" * 60)
    print("测试 5: 情绪逃逸（对话轮数 >= 6 触发兜底）")
    print("=" * 60)

    session_id = "test-escalation"

    # 模拟 3 轮对话（6 条消息）
    queries = [
        "你好，我想问一下医保问题",
        "具体是怎么报销的？",
        "需要哪些材料？",
        "在哪里办理？",
        "可以网上办理吗？",
        "那电话是多少？",
    ]

    for i, q in enumerate(queries, 1):
        print(f"\n📝 第{i}轮: {q}")
        result = await ainvoke(q, session_id=session_id)
        print(f"   置信度: {result.get('confidence_score', 0):.2f}, "
              f"工单: {result.get('work_order_id', '无')}")

        # 如果已经触发兜底，停止
        if result.get("work_order_id"):
            print(f"\n⚠️ 已触发兜底工单！")
            break


async def test_direct_graph():
    """测试 6: 直接使用 gov_agent_app"""
    print("\n" + "=" * 60)
    print("测试 6: 直接使用 gov_agent_app")
    print("=" * 60)

    from langchain_core.messages import HumanMessage

    session_id = "test-direct"
    config = {"configurable": {"thread_id": session_id}}

    # 使用 invoke 方法
    inputs = {"messages": [HumanMessage(content="请问低保怎么办理？")]}

    result = gov_agent_app.invoke(inputs, config=config)

    print(f"\n📋 结果:")
    print(f"   - 回复: {result.get('final_reply', '')[:100]}...")
    print(f"   - 置信度: {result.get('confidence_score', 0):.2f}")


async def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("🚀 LangGraph Agent 测试 Demo")
    print("=" * 60)

    # 选择要运行的测试
    tests = [
        ("正常咨询流程", test_normal_consult),
        ("多轮对话", test_multi_turn),
        ("投诉举报", test_complaint),
        ("低置信度", test_low_confidence),
        ("情绪逃逸", test_escalation),
        ("直接使用 Graph", test_direct_graph),
    ]

    # 运行所有测试
    for name, test_func in tests:
        try:
            await test_func()
        except Exception as e:
            print(f"\n❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("✅ 所有测试完成")
    print("=" * 60)


if __name__ == "__main__":
    # 设置日志级别
    import logging
    logging.getLogger("src").setLevel(logging.WARNING)

    asyncio.run(main())