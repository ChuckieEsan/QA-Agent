"""
政务智能体编排层 (ReAct Agent)
"""

from pathlib import Path
from typing import List, TypedDict, Annotated, Literal
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from src.app.agents.tools.registry import get_all_tools
from src.app.infra.utils.logger import get_logger
from src.app.infra.llm import create_llm_service

logger = get_logger(__name__)

# 从外置文件读取系统提示词
def _load_system_prompt() -> str:
    """从 prompts 目录加载系统提示词"""
    prompt_path = Path(__file__).parent.parent / "prompts" / "agent_system_prompt.md"
    return prompt_path.read_text(encoding="utf-8")


system_prompt = _load_system_prompt()

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
    
    def route_after_tools(state: AgentState) -> Literal["agent", "__end__"]:
        last_message = state["messages"][-1]
        
        # 核心判断：如果最后一条消息是 AIMessage，
        # 说明我们的 `validate_answer_tool` 熔断生效，成功把草稿塞进来了
        # 既然回答已经就绪，图必须立刻结束，绝不能再去找大模型！
        if isinstance(last_message, AIMessage):
            return "__end__"
            
        # 否则（通常是普通的 ToolMessage 返回了检索结果），需要大模型继续思考
        return "agent"


    workflow = StateGraph(AgentState)
    
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", execute_tools)
    
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    
    # workflow.add_edge("tools", "agent")
    workflow.add_conditional_edges("tools", route_after_tools)
    
    memory = MemorySaver()
    agent_app = workflow.compile(checkpointer=memory)
    
    logger.info("自定义编排构建完成！")
    return agent_app


gov_agent_app = create_gov_agent()