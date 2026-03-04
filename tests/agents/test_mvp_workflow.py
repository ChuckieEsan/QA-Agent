"""
LangGraph MVP 工作流测试
"""

import asyncio
from src.app.agents import invoke, ainvoke
from src.app.infra.utils.logger import get_logger

logger = get_logger(__name__)


def test_sync_invoke():
    """测试同步调用"""
    print("=" * 60)
    print("测试同步调用 LangGraph MVP 工作流")
    print("=" * 60)

    query = "泸州雨露计划补贴标准是多少？"
    print(f"\n📝 查询：{query}")

    try:
        result = invoke(query)
        print("\n✅ 调用成功!")
        print(f"\n📋 最终回复：{result.get('final_response', 'N/A')[:200]}...")
        print(f"\n📊 状态信息:")
        print(f"   - 诉求类型：{result.get('appeal_type', 'N/A')}")
        print(f"   - 紧急程度：{result.get('urgency_level', 'N/A')}")
        print(f"   - 办理部门：{result.get('department', 'N/A')}")
        print(f"   - 检索结果数：{len(result.get('retrieval_results', []))}")
        return True
    except Exception as e:
        print(f"\n❌ 调用失败：{e}")
        return False


async def test_async_invoke():
    """测试异步调用"""
    print("=" * 60)
    print("测试异步调用 LangGraph MVP 工作流")
    print("=" * 60)

    query = "泸州雨露计划补贴标准是多少？"
    print(f"\n📝 查询：{query}")

    try:
        result = await ainvoke(query)
        print("\n✅ 调用成功!")
        print(f"\n📋 最终回复：{result.get('final_response', 'N/A')[:200]}...")
        print(f"\n📊 状态信息:")
        print(f"   - 诉求类型：{result.get('appeal_type', 'N/A')}")
        print(f"   - 紧急程度：{result.get('urgency_level', 'N/A')}")
        print(f"   - 办理部门：{result.get('department', 'N/A')}")
        print(f"   - 检索结果数：{len(result.get('retrieval_results', []))}")
        return True
    except Exception as e:
        print(f"\n❌ 调用失败：{e}")
        return False


def test_nodes():
    """测试各个 Node"""
    print("=" * 60)
    print("测试各个 Node")
    print("=" * 60)

    from src.app.agents.nodes import (
        preprocess_query,
        classify_appeal,
        retrieve_context,
    )

    # 测试预处理节点
    print("\n1️⃣ 测试预处理节点...")
    state = {"raw_query": "张三的电话是 13812341234，住在泸州市江阳区 1 号楼 2 单元 3 号"}
    result = preprocess_query(state)
    print(f"   清洗后：{result.get('cleaned_query', 'N/A')}")
    print(f"   脱敏后：{result.get('desensitized_query', 'N/A')}")
    print(f"   提取要素：{result.get('extracted_elements', {})}")

    # 测试分类节点
    print("\n2️⃣ 测试分类节点...")
    state = {"cleaned_query": "泸州雨露计划补贴标准是多少？"}
    result = classify_appeal(state)
    print(f"   诉求类型：{result.get('appeal_type', 'N/A')}")
    print(f"   紧急程度：{result.get('urgency_level', 'N/A')}")
    print(f"   办理部门：{result.get('department', 'N/A')}")

    # 测试检索节点
    print("\n3️⃣ 测试检索节点...")
    state = {"cleaned_query": "泸州雨露计划补贴标准是多少？"}
    result = retrieve_context(state)
    results = result.get("retrieval_results", [])
    print(f"   检索结果数：{len(results)}")
    if results:
        print(f"   第一个结果：{results[0].get('title', 'N/A')}")

    print("\n✅ Node 测试完成")


if __name__ == "__main__":
    # 设置日志级别
    import logging
    logging.getLogger("src").setLevel(logging.INFO)

    # 测试 Node
    test_nodes()

    # 测试同步调用
    # test_sync_invoke()

    # 测试异步调用
    # asyncio.run(test_async_invoke())

    print("\n" + "=" * 60)
    print("所有测试完成")
    print("=" * 60)
