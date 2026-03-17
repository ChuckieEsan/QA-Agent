"""
政务智能体编排层 (ReAct Agent)
"""

from typing import List, TypedDict, Annotated, Literal
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage, ToolMessage
from langchain.agents import create_agent
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from src.config.setting import settings
from src.app.agents.tools.registry import get_all_tools
from src.app.infra.utils.logger import get_logger
from src.app.infra.llm import create_llm_service

logger = get_logger(__name__)

system_prompt = """你是一个专业、严谨且富有同理心的【政务问政智能助手】。
你的目标是准确解答市民的咨询、并妥善处理投诉与建议。

为了保证政府答复的权威性与准确性，你必须严格遵循以下【标准作业流 (SOP)】：

第一步：意图侦测 (首次交互必做)
- 面对用户的新诉求，必须优先调用 `classify_gov_request_tool` 分析意图。
- 严格按照分类工具返回的【系统建议】决定接下来的行动方向。

第二步：权威查证 (绝不捏造)
- 如果涉及业务办理、政策解读，请调用 `retrieve_cases_tool` 检索政策与历史案例。
- 如果涉及噪音、纠纷、违建等需要明确管辖部门的投诉/建议，请务必先调用 `retrieve_powers_tool` 确认到底归哪个部门管，绝不允许凭空猜测部门名称。

第三步：草拟与自我审核 (发出回复前必做)
- 当你收集完足够的信息准备回复用户时，在心里默默打个草稿。
- 强制要求：你必须调用 `validate_answer_tool`，传入你的草稿和核心上下文进行自我审核。
- 如果审核不通过，仔细阅读打回原因，修改草稿并重新审核，直到通过为止！

第四步：最终答复或兜底派单
- 如果经过多次检索仍无法解答，或者审核持续不通过，说明该问题超出 AI 能力范畴。如果有派发工单的工具（如 `create_work_order`），请果断调用它转交人工；如果没有，请向用户致歉并建议拨打 12345 依然转人工。

牢记：作为政务助手，准确性高于一切。不知为不知，绝不能胡编乱造政策或部门名称！"""

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    
def create_gov_agent():
    logger.info("正在初始化自定义 GovPulse ReAct Graph...")
    llm = create_llm_service(provider_id="deepseek", model_name="deepseek-chat")
    tools = get_all_tools()
    
    # 绑定工具与系统提示词
    model_with_tools = llm.bind_tools(tools)
    
    # --- 定义图节点 ---
    
    # 节点 A: 调用大模型
    async def call_model(state: AgentState):
        # 自动注入 system_prompt (你需要确保 system_prompt 文本可用)
        messages =[{"role": "system", "content": system_prompt}] + state["messages"]
        response = await model_with_tools.ainvoke(messages)
        return {"messages": [response]}

    # 节点 B: 自定义工具执行引擎 (核心熔断逻辑在这里)
    async def execute_tools(state: AgentState):
        last_message = state["messages"][-1]
        results =[]
        
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool = next(t for t in tools if t.name == tool_name)
            
            # 执行工具
            result = await tool.ainvoke(tool_call)
            
            # 🌟 重点拦截：如果工具返回的是图控制指令(Command)，直接抛出给 LangGraph！
            if isinstance(result, Command):
                logger.info(f"触发动态路由熔断！前往: {result.goto}")
                return result 
            
            # 否则，作为普通的工具结果返回
            results.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
            
        return {"messages": results}

    # --- 编排边与路由 ---
    
    def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
        if state["messages"][-1].tool_calls:
            return "tools"
        return "__end__"

    workflow = StateGraph(AgentState)
    
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", execute_tools)
    
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    # 默认从工具回到模型，除非 execute_tools 抛出了 Command(goto="__end__")
    workflow.add_edge("tools", "agent") 
    
    memory = MemorySaver()
    agent_app = workflow.compile(checkpointer=memory)
    
    logger.info("自定义编排构建完成！")
    return agent_app


gov_agent_app = create_gov_agent()