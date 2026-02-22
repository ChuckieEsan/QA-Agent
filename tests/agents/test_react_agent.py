"""
ReAct Agent 测试用例
"""

import asyncio
import pytest
from src.app.agents import ReactAgent, ReactStep, ToolRegistry
from src.app.agents.tools import BaseTool


class TestReactTools:
    """测试 ReAct 工具"""

    @pytest.mark.asyncio
    async def test_create_default_tools(self):
        """测试工具集创建"""
        tools = {
    "retrieve": ToolRegistry.get_instance("retrieve"),
    "generate": ToolRegistry.get_instance("generate"),
    "classify": ToolRegistry.get_instance("classify"),
    "validate": ToolRegistry.get_instance("validate"),
}

        assert "retrieve" in tools
        assert "generate" in tools
        assert "classify" in tools
        assert "validate" in tools

        assert isinstance(tools["retrieve"], BaseTool)
        assert isinstance(tools["generate"], BaseTool)
        assert isinstance(tools["classify"], BaseTool)
        assert isinstance(tools["validate"], BaseTool)

        print("✅ 工具集创建成功")

    @pytest.mark.asyncio
    async def test_retrieval_tool(self):
        """测试检索工具"""
        from src.app.agents.tools import RetrievalTool

        tool = RetrievalTool()

        result = await tool.execute(
            query="2024年泸州雨露计划补贴标准",
            top_k=2
        )

        assert "context" in result
        assert "results" in result
        assert "metadata" in result

        print("✅ 检索工具测试通过")

    @pytest.mark.asyncio
    async def test_generation_tool(self):
        """测试生成工具"""
        from src.app.agents.tools import GenerationTool

        tool = GenerationTool()

        result = await tool.execute(
            prompt="你好，请问有什么可以帮助您的？"
        )

        assert "answer" in result
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

        print("✅ 生成工具测试通过")

    @pytest.mark.asyncio
    async def test_classification_tool(self):
        """测试分类工具"""
        from src.app.agents.tools import ClassificationTool

        tool = ClassificationTool()

        result = await tool.execute(
            query="2024年泸州雨露计划补贴标准"
        )

        assert "type" in result
        assert "confidence" in result
        assert isinstance(result["type"], str)
        assert isinstance(result["confidence"], float)

        print("✅ 分类工具测试通过")


class TestReactAgent:
    """测试 ReactAgent"""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """测试初始化"""
        tools = {
    "retrieve": ToolRegistry.get_instance("retrieve"),
    "generate": ToolRegistry.get_instance("generate"),
    "classify": ToolRegistry.get_instance("classify"),
    "validate": ToolRegistry.get_instance("validate"),
}
        agent = ReactAgent(tools, max_steps=3)

        assert agent.max_steps == 3
        assert len(agent.tools) == 4

        print("✅ ReactAgent 初始化成功")

    @pytest.mark.asyncio
    async def test_basic_reasoning(self):
        """测试基本推理"""
        tools = {
    "retrieve": ToolRegistry.get_instance("retrieve"),
    "generate": ToolRegistry.get_instance("generate"),
    "classify": ToolRegistry.get_instance("classify"),
    "validate": ToolRegistry.get_instance("validate"),
}
        agent = ReactAgent(tools, max_steps=3)

        result = await agent.process("2024年泸州雨露计划补贴标准")

        # 验证输出格式
        assert "answer" in result
        assert "steps_history" in result
        assert "steps_count" in result

        # 验证步数限制
        assert result["steps_count"] <= 3

        # 验证推理历史
        assert len(result["steps_history"]) == result["steps_count"]
        assert all("thought" in step for step in result["steps_history"])
        assert all("action" in step for step in result["steps_history"])
        assert all("observation" in step for step in result["steps_history"])

        print("✅ 基础推理测试通过")
        print(f"  推理步数: {result['steps_count']}")
        print(f"  答案: {result['answer'][:50]}...")

    @pytest.mark.asyncio
    async def test_multi_step_reasoning(self):
        """测试多步推理"""
        tools = {
    "retrieve": ToolRegistry.get_instance("retrieve"),
    "generate": ToolRegistry.get_instance("generate"),
    "classify": ToolRegistry.get_instance("classify"),
    "validate": ToolRegistry.get_instance("validate"),
}
        agent = ReactAgent(tools, max_steps=5)

        result = await agent.process("泸州小微企业有哪些税收优惠政策？")

        # 验证步数
        assert 1 <= result["steps_count"] <= 5

        # 验证答案
        assert len(result["answer"]) > 0

        print("✅ 多步推理测试通过")
        print(f"  推理步数: {result['steps_count']}")

    @pytest.mark.asyncio
    async def test_step_history_detail(self):
        """测试推理历史详细信息"""
        tools = {
    "retrieve": ToolRegistry.get_instance("retrieve"),
    "generate": ToolRegistry.get_instance("generate"),
    "classify": ToolRegistry.get_instance("classify"),
    "validate": ToolRegistry.get_instance("validate"),
}
        agent = ReactAgent(tools, max_steps=3)

        result = await agent.process("如何办理身份证？")

        # 检查第一步
        first_step = result["steps_history"][0]
        assert "step_number" in first_step
        assert "thought" in first_step
        assert "action" in first_step
        assert "action_input" in first_step
        assert "observation" in first_step
        assert "timestamp" in first_step

        print("✅ 推理历史详细信息测试通过")


class TestToolRegistry:
    """测试工具注册表"""

    def test_register_tool(self):
        """测试注册工具"""

        @ToolRegistry.register("echo")
        class EchoTool:
            name = "echo"
            description = "回显测试"

            async def execute(self, text: str = "") -> dict:
                return {"result": f"Echo: {text}"}

        assert "echo" in ToolRegistry.list_all()
        tool = ToolRegistry.create("echo")
        assert tool is not None

    def test_list_all_tools(self):
        """测试列出所有工具"""
        tools = ToolRegistry.list_all()
        assert isinstance(tools, dict)

    def test_get_nonexistent_tool(self):
        """测试获取不存在的工具"""
        with pytest.raises(KeyError):
            ToolRegistry.get("nonexistent_tool")


# ==================== 手动测试脚本 ====================

async def manual_test_basic_functionality():
    """手动测试基础功能"""
    print("\n=== 手动测试：基础功能 ===")

    # 测试工具集创建
    tools = {
    "retrieve": ToolRegistry.get_instance("retrieve"),
    "generate": ToolRegistry.get_instance("generate"),
    "classify": ToolRegistry.get_instance("classify"),
    "validate": ToolRegistry.get_instance("validate"),
}
    print(f"✅ 工具集创建成功: {list(tools.keys())}")

    # 测试 ReactAgent
    agent = ReactAgent(tools, max_steps=3)
    print(f"✅ ReactAgent 初始化成功 (max_steps={agent.max_steps})")

    # 测试基本推理
    result = await agent.process("2024年泸州雨露计划补贴标准")

    print(f"✅ 基础功能测试通过")
    print(f"  推理步数: {result['steps_count']}")
    print(f"  答案: {result['answer'][:100]}...")
    print(f"  步骤历史: {len(result['steps_history'])} 步")


async def manual_test_tool_extension():
    """手动测试工具扩展"""
    print("\n=== 手动测试：工具扩展 ===")

    # 注册自定义工具
    @ToolRegistry.register("calculator")
    class CalculatorTool:
        name = "calculator"
        description = "计算数学表达式"

        async def execute(self, expression: str = "") -> dict:
            try:
                result = eval(expression)
                return {"result": result}
            except:
                return {"result": "计算失败"}

    # 创建工具集并添加自定义工具（自定义工具已自动注册并创建实例）
    tools = {
    "retrieve": ToolRegistry.get_instance("retrieve"),
    "generate": ToolRegistry.get_instance("generate"),
    "classify": ToolRegistry.get_instance("classify"),
    "validate": ToolRegistry.get_instance("validate"),
    "calculator": ToolRegistry.get_instance("calculator"),
    }

    agent = ReactAgent(tools, max_steps=3)
    print(f"✅ 自定义工具注册成功")

    result = await agent.process("计算 2+2*5")
    print(f"✅ 工具扩展测试通过")
    print(f"  最终答案: {result['answer']}")


async def manual_test_full_workflow():
    """手动测试完整工作流"""
    print("\n=== 手动测试：完整工作流 ===")

    tools = {
    "retrieve": ToolRegistry.get_instance("retrieve"),
    "generate": ToolRegistry.get_instance("generate"),
    "classify": ToolRegistry.get_instance("classify"),
    "validate": ToolRegistry.get_instance("validate"),
}
    agent = ReactAgent(tools, max_steps=5)

    # 测试不同复杂度的问题
    test_cases = [
        "如何办理身份证？",
        "2024年泸州雨露计划补贴标准",
        "泸州小微企业税收优惠和社保补贴政策",
    ]

    for query in test_cases:
        print(f"\n--- 测试: {query} ---")
        result = await agent.process(query)

        print(f"  答案: {result['answer'][:100]}...")
        print(f"  推理步数: {result['steps_count']}")
        print(f"  检索来源: {len(result['sources'])} 个")

        # 检查推理历史
        assert len(result['steps_history']) == result['steps_count']

        # 打印推理步骤
        print(f"  推理步骤:")
        for step in result['steps_history']:
            print(f"    Step {step['step_number']}: {step['action']}")

    print("\n✅ 完整工作流测试通过")


if __name__ == "__main__":
    # 运行手动测试
    asyncio.run(manual_test_basic_functionality())
    asyncio.run(manual_test_tool_extension())
    asyncio.run(manual_test_full_workflow())

    print("\n" + "=" * 50)
    print("✅ 所有手动测试通过！")
    print("=" * 50)
