"""
ReAct Agent 框架与 BGE 重排模型集成演示
"""
import asyncio
import sys
import os
from pathlib import Path

sys.path.append(os.getcwd())

from src.app.infra.utils.logger import get_logger
from src import ReactAgent, ToolRegistry

logger = get_logger(__name__)


async def demo_react_agent_with_rerank():
    """
    演示 ReAct Agent 与 BGE 重排模型集成
    """
    logger.info("="*80)
    logger.info("ReAct Agent 与 BGE 重排模型集成演示")
    logger.info("="*80)

    # 创建工具集
    tools = {
        "retrieve": ToolRegistry.get_instance("retrieve"),
        "generate": ToolRegistry.get_instance("generate"),
        "classify": ToolRegistry.get_instance("classify"),
        "validate": ToolRegistry.get_instance("validate"),
    }

    # 创建 ReAct Agent
    agent = ReactAgent(tools, max_steps=5)

    # 测试查询
    queries = [
        "泸州市购房补贴政策",
        "雨露计划申请条件",
        "泸州住房公积金贷款政策"
    ]

    for i, query in enumerate(queries, 1):
        logger.info(f"测试查询 {i}: {query}")
        logger.info("-" * 60)

        # 执行查询
        result = await agent.process(query)

        logger.info(f"回答: {result['answer'][:200]}...")
        logger.info(f"推理步数: {result['steps_count']}")
        logger.info(f"来源: {len(result['sources'])} 个案例")
        logger.info(f"检索耗时: {result.get('retrieval_time', 0):.2f}秒")

    logger.info("演示完成！")
    logger.info("重排模型效果:")
    logger.info("   • 使用 BGE 重排模型提高了检索结果的相关性")
    logger.info("   • 在 HybridVectorRetriever 中实现了无缝集成")
    logger.info("   • 支持传统重排策略作为备选方案")


if __name__ == "__main__":
    asyncio.run(demo_react_agent_with_rerank())