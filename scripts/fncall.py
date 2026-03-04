from typing import List, Iterator
from dataclasses import dataclass
from enum import Enum

from src.app.infra.utils.logger import get_logger
logger = get_logger(__name__)

# 模拟 Message 类
@dataclass
class Message:
    role: str
    content: str
    name: str = None

# 模拟 FnCallAgent 的简化版
class SimpleFnCallAgent:
    def _run(self, messages: List[Message]) -> Iterator[List[Message]]:
        """模拟 Qwen 的 _run 方法"""
        response = []  # 累计的响应
        
        # 模拟循环过程
        for step in range(3):  # 假设最多3步
            logger.info(f"第 {step+1} 次循环开始")
            
            # 模拟 LLM 流式输出
            llm_output_stream = self._simulate_llm_stream(step)
            
            # 处理流式输出
            current_step_messages = []
            for chunk in llm_output_stream:
                current_step_messages.append(chunk)
                logger.debug(f"生成器产生: response + chunk = {response + current_step_messages}")
                yield response + current_step_messages
            
            # 将当前步骤的消息添加到累计响应
            response.extend(current_step_messages)
            
            # 检查是否需要工具调用
            if step == 1:  # 假设第二步需要工具调用
                tool_result = Message(role="function", name="calculator", content="42")
                response.append(tool_result)
                logger.debug(f"工具调用后: {response}")
                yield response  # 返回包含工具结果的状态
        
        logger.info("最终返回")
        yield response
    
    def _simulate_llm_stream(self, step: int) -> List[Message]:
        """模拟 LLM 流式输出，每次产生一个 Message 列表"""
        if step == 0:
            return [Message(role="assistant", content="让我计算一下...")]
        elif step == 1:
            return [Message(role="assistant", content="调用计算器", name="calculator")]
        else:
            return [Message(role="assistant", content="最终答案是42")]

# 使用示例
agent = SimpleFnCallAgent()
messages = [Message(role="user", content="40 + 2 等于多少？")]

logger.info("开始执行 agent._run()...")
logger.info("-" * 50)

# 遍历生成器的每次产出
for i, state_snapshot in enumerate(agent._run(messages), 1):
    logger.debug(f"【第{i}次yield返回】")
    logger.debug(f"完整状态快照（{len(state_snapshot)}条消息）:")
    # for msg in state_snapshot:
    #     print(f"  - {msg.role}: {msg.content} (name: {msg.name})")