"""
RAG Agent
使用所有组件实现完整的 RAG 功能

新增能力：
- Agent 意图分析：分析用户查询并生成检索决策
- 动态策略调整：根据决策结果调整检索策略
"""

import asyncio
import json
import traceback
from typing import Dict, List, Optional, Tuple
from dashscope import Generation
from src.app.agents.base_agent import BaseAgent
from src.app.components.retrievers import BaseRetriever, HybridVectorRetriever
from src.app.components.generators import BaseGenerator, LLMGenerator
from src.app.components.classifier import BaseClassifier, GovClassifier
from src.app.components.memory import BaseMemory, ConversationMemory
from src.app.components.quality.answer_validator import AnswerValidator
from src.app.agents.models.agent_decision import (
    AgentDecision,
    AgentDecisionType,
    RetrievalStrategy
)
from src.app.infra.utils.logger import get_logger
from src.app.infra.llm.multi_model_service import get_light_llm_service

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
        memory: Optional[BaseMemory] = None,
        validator: Optional[AnswerValidator] = None
    ):
        super().__init__(name="RagAgent")

        # 依赖注入：所有组件
        self.retriever = retriever or HybridVectorRetriever()
        self.generator = generator or LLMGenerator()
        self.classifier = classifier or GovClassifier()
        self.memory = memory or ConversationMemory()
        self.validator = validator or AnswerValidator()

        logger.info("✅ RagAgent 初始化完成，组件加载：")
        logger.info(f"  - Retriever: {type(self.retriever).__name__}")
        logger.info(f"  - Generator: {type(self.generator).__name__}")
        logger.info(f"  - Classifier: {type(self.classifier).__name__}")
        logger.info(f"  - Memory: {type(self.memory).__name__}")
        logger.info(f"  - Validator: {type(self.validator).__name__}")

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
                "decision": Dict,           # 新增：Agent决策
                "sources": List[Dict],
                "metadata": Dict,
                "quality_check": Dict
            }
        """
        # 步骤 0：分析查询意图（Agent决策）
        logger.info(f"📋 [Agent Decision] 正在分析查询意图...")
        decision = await self.analyze_intent(query, history)
        logger.info(f"  → 决策类型: {decision.decision_type} | 意图: {decision.intent[:30]}")

        # 步骤 1：分类问政类型
        logger.info(f"📋 [Classifier] 正在分类问政类型...")
        classification = await self.classifier.classify_gov_request(query)
        logger.info(f"  → 问政类型: {classification['type']} (置信度: {classification['confidence']:.2f})")

        # 步骤 2：检索相关案例
        logger.info(f"🔍 [Retriever] 执行检索...")
        context, results, metadata = self.retriever.retrieve(query)
        logger.info(f"  → 检索到 {len(results)} 个结果")

        # 步骤 3：生成回答
        logger.info("🤖 [Generator] 正在生成回答...")
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
        logger.info("🔍 [Validator] 正在校验回答质量...")
        validation = await self.validator.validate(answer, query, context)

        return {
            "query": query,
            "answer": answer,
            "classification": classification,
            "decision": decision.model_dump(),  # 新增：Agent决策
            "sources": results[:5],  # 最多返回 5 个来源
            "metadata": metadata,
            "quality_check": validation
        }

    async def initialize(self) -> None:
        """初始化所有组件"""
        # 并行初始化所有组件
        await asyncio.gather(
            self.classifier.initialize(),
            self.generator.initialize(),
            self.validator.initialize()
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
                "validator": type(self.validator).__name__,
            },
            "memory_stats": memory_stats
        }

    # ==================== Agent 决策能力 ====================

    async def analyze_intent(
        self,
        query: str,
        history: Optional[List[Dict]] = None
    ) -> AgentDecision:
        """
        Agent 核心能力：分析查询意图并生成检索决策

        Args:
            query: 用户查询
            history: 对话历史（可选）

        Returns:
            AgentDecision: 决策结果
        """
        # 构建决策 Prompt
        prompt_parts = [
            "你是一名 RAG Agent 决策助手，负责分析用户查询并给出检索决策。",
            "",
            "# 决策任务：",
            "1. 分析用户查询的核心意图",
            "2. 判断是否需要检索知识库",
            "3. 选择最优检索策略",
            "4. 调整检索参数（如 top_k、阈值）",
            "5. 必要时重写查询语句",
            "",
            "# 决策规则：",
            "- direct_answer：通用政务常识、无需具体案例支撑的问题（如'如何办理身份证'的通用流程）",
            "- need_retrieval：需要具体政策/案例支撑的问题（如'2024年泸州雨露计划补贴标准'）",
            "- multi_retrieval：跨部门/多政策的复杂问题（如'泸州小微企业税收优惠+社保补贴'）",
            "- cannot_answer：非泸州市政务问题/无意义问题/敏感问题",
            "",
            "# 检索策略选择：",
            "- hybrid：默认策略，混合语义+关键词检索",
            "- keyword：强关键词特征的问题（如'2024年泸州医保缴费标准'）",
            "- semantic_only：语义模糊/多义词问题（如'泸州创业扶持政策'）",
            "- cross_dept：跨部门问题（如'泸州住房补贴+公积金政策'）",
            "",
            "# 输出格式（JSON）：",
            "{",
            '    "decision_type": "direct_answer|need_retrieval|multi_retrieval|cannot_answer",',
            '    "retrieval_strategy": "hybrid|keyword|semantic_only|cross_dept",',
            '    "retrieval_params": {"top_k": 5-10, "threshold": 0.5-0.8},',
            '    "query_rewritten": "重写后的查询语句（可选）",',
            '    "intent": "核心意图描述",',
            '    "confidence": 0.0-1.0',
            "}",
            "",
            "# 注意：",
            "- retrieval_strategy/cross_dept 仅在 decision_type 为 need_retrieval/multi_retrieval 时必填",
            "- retrieval_params 需根据问题复杂度调整（复杂问题 top_k=8-10，简单问题=3-5）",
            "- query_rewritten 需更精准表达核心意图（如原问题'雨露计划多少钱'→'2024年泸州市雨露计划补贴金额标准'）",
            "",
            "# 用户查询：",
            query,
        ]

        # 添加对话历史
        if history and len(history) > 0:
            prompt_parts.append("\n# 对话历史：")
            for turn in history[-3:]:
                role = "用户" if turn["role"] == "user" else "助手"
                prompt_parts.append(f"{role}：{turn['content']}")

        prompt = "\n".join(prompt_parts)

        try:
            # 使用轻量模型生成决策
            light_llm = get_light_llm_service()
            response = Generation.call(
                model=light_llm.get_model_name(),
                prompt=prompt,
                temperature=light_llm.get_config().temperature,
                max_tokens=500,
                top_p=light_llm.get_config().top_p,
                result_format='text'
            )

            if response.status_code == 200:
                decision_str = response.output.text
                # 解析 JSON 决策
                decision_data = json.loads(decision_str)
                return AgentDecision(**decision_data)
            else:
                raise Exception(f"决策生成失败: {response.code} - {response.message}")

        except Exception as e:
            logger.error(f"❌ Agent 决策失败: {e}")
            logger.error(traceback.format_exc())
            # 返回默认决策（兜底）
            return AgentDecision(
                decision_type="need_retrieval",
                retrieval_strategy="hybrid",
                retrieval_params={"top_k": 5, "threshold": 0.6},
                query_rewritten=query,
                intent=f"无法解析意图：通用查询",
                confidence=0.5
            )


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
