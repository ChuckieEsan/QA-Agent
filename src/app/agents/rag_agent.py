"""
RAG Agent
使用所有组件实现完整的 RAG 功能
"""

import asyncio
from typing import Dict, List, Optional, Tuple
from src.app.agents.base_agent import BaseAgent
from src.app.components.retrievers import BaseRetriever, get_retriever_instance
from src.app.components.generators import BaseGenerator, LLMGenerator
from src.app.components.classifier import BaseClassifier, GovClassifier
from src.app.components.memory import BaseMemory, ConversationMemory
from src.app.infra.utils.logger import get_logger

logger = get_logger(__name__)


class RagAgent(BaseAgent):
    """
    RAG Agent

    使用所有组件实现完整功能：
    - Generator：生成回答
    - Classifier：分类问政类型
    - Memory：管理对话历史
    - Retriever：检索相关案例
    """

    def __init__(
        self,
        retriever: Optional[BaseRetriever] = None,
        generator: Optional[BaseGenerator] = None,
        classifier: Optional[BaseClassifier] = None,
        memory: Optional[BaseMemory] = None
    ):
        super().__init__(name="RagAgent")

        # 依赖注入：所有组件
        self.retriever = retriever or get_retriever_instance()
        self.generator = generator or LLMGenerator()
        self.classifier = classifier or GovClassifier()
        self.memory = memory or ConversationMemory()

        logger.info("✅ RagAgent 初始化完成，组件加载：")
        logger.info(f"  - Retriever: {type(self.retriever).__name__}")
        logger.info(f"  - Generator: {type(self.generator).__name__}")
        logger.info(f"  - Classifier: {type(self.classifier).__name__}")
        logger.info(f"  - Memory: {type(self.memory).__name__}")

    async def process(
        self,
        query: str,
        history: Optional[List[Dict]] = None,
        **kwargs
    ) -> Dict[str, any]:
        """
        处理查询（实现 BaseAgent）

        Args:
            query: 用户查询
            history: 对话历史（可选）
            **kwargs: 其他参数

        Returns:
            {
                "query": str,
                "answer": str,
                "classification": Dict,
                "sources": List[Dict],
                "metadata": Dict,
                "quality_check": Dict
            }
        """
        # 步骤 1：分类问政类型
        logger.info(f"📋 开始处理查询: {query[:30]}...")
        classification = await self.classifier.classify_gov_request(query)
        logger.info(f"  → 问政类型: {classification['type']} (置信度: {classification['confidence']:.2f})")

        # 步骤 2：检索相关案例
        logger.info("🔍 执行检索...")
        context, results, metadata = self.retriever.retrieve(query)
        logger.info(f"  → 检索到 {len(results)} 个结果")

        # 步骤 3：生成回答
        logger.info("🤖 生成回答...")
        answer = await self.generator.generate(
            prompt=query,
            system_message="基于检索到的案例生成准确回答",
            history=history
        )
        logger.info(f"  → 回答生成完成: {answer[:50]}...")

        # 步骤 4：保存到记忆
        if history is None:
            self.memory.add_message({"role": "user", "content": query})
            self.memory.add_message({"role": "assistant", "content": answer})
        else:
            # 使用外部历史
            pass

        # 步骤 5：质量校验
        validation = await self.generator.generate_with_validation(
            prompt=query,
            validation_criteria={}
        )

        return {
            "query": query,
            "answer": answer,
            "classification": classification,
            "sources": results[:5],  # 最多返回 5 个来源
            "metadata": metadata,
            "quality_check": validation
        }

    async def initialize(self) -> None:
        """初始化所有组件"""
        # 并行初始化所有组件
        await asyncio.gather(
            self.classifier.initialize(),
            self.generator.initialize()
        )
        self._initialized = True
        logger.info("✅ RagAgent 初始化完成")

    def get_status(self) -> Dict[str, any]:
        """获取状态"""
        memory_stats = self.memory.get_stats()
        return {
            "name": self.name,
            "initialized": self._initialized,
            "created_at": self.created_at.isoformat(),
            "components": {
                "retriever": type(self.retriever).__name__,
                "generator": type(self.generator).__name__,
                "classifier": type(self.classifier).__name__,
                "memory": type(self.memory).__name__,
            },
            "memory_stats": memory_stats
        }

    # ==================== 兼容性方法 ====================

    async def query(
        self,
        query: str,
        history: Optional[List[Dict]] = None,
        **kwargs
    ) -> Dict[str, any]:
        """
        兼容旧接口：query() 方法

        Args:
            query: 用户查询
            history: 对话历史（可选）
            **kwargs: 其他参数

        Returns:
            处理结果
        """
        return await self.process(query, history, **kwargs)


# ==================== 工具函数 ====================

async def query_agentic_rag(
    query: str,
    history: Optional[List[Dict]] = None,
    **kwargs
) -> Dict[str, any]:
    """
    快速调用 Agentic RAG 查询

    Args:
        query: 用户查询
        history: 对话历史（可选）
        **kwargs: 其他参数

    Returns:
        处理结果
    """
    agent = RagAgent()
    return await agent.process(query, history, **kwargs)
