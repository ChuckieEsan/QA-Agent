"""
Agentic RAG协调引擎 - 核心决策与流程控制
增强能力：Agent决策驱动、多策略检索、结果评估、自动重检索
"""

import time
from typing import Dict, List, Optional, Tuple
from app.services.retriever import HybridVectorRetriever
from app.services.llm_service import LLMService, AgentDecision, get_llm_service
from app.core.logger import get_logger

logger = get_logger(__name__)


class AgenticRAGEngine:
    """
    Agentic RAG协调引擎
    核心流程：
    1. Agent意图分析与决策
    2. 基于决策的多策略检索
    3. 检索结果质量评估
    4. 智能生成与质量校验
    5. 自动重检索（可选）
    """

    def __init__(self):
        self.retriever = HybridVectorRetriever()
        self.llm_service: LLMService = get_llm_service()
        self.min_quality_score = 0.7  # 最低回答质量阈值
        logger.info("🤖 Agentic RAG引擎初始化完成")

    async def _evaluate_retrieval_quality(
        self, query: str, results: List[Dict], metadata: Dict
    ) -> Dict[str, any]:
        """
        Agent核心能力：评估检索结果质量
        检查：相关性、覆盖度、权威性
        """
        if not results:
            return {
                "retrieval_quality_score": 0.0,
                "suggestion": "无检索结果，建议扩大检索范围",
                "need_reretrieval": True,
            }

        # 计算核心指标
        avg_similarity = metadata.get("avg_similarity", 0.0)
        num_results = metadata.get("num_results", 0)
        dept_coverage = len(set([r["entity"].get("department", "") for r in results]))

        # 综合检索质量评分
        retrieval_score = (
            avg_similarity * 0.6  # 相似度权重60%
            + min(1.0, num_results / 10) * 0.2  # 数量权重20%
            + min(1.0, dept_coverage / 5) * 0.2  # 部门覆盖度权重20%
        )

        # 判断是否需要重检索
        need_reretrieval = retrieval_score < 0.6 or (  # 低质量检索结果
            num_results < 2 and avg_similarity < 0.7
        )  # 结果少且相似度低

        return {
            "retrieval_quality_score": retrieval_score,
            "avg_similarity": avg_similarity,
            "num_results": num_results,
            "dept_coverage": dept_coverage,
            "suggestion": "需要重检索" if need_reretrieval else "检索结果合格",
            "need_reretrieval": need_reretrieval,
        }

    async def _reretrieve_with_adjusted_params(
        self, query: str, decision: AgentDecision
    ) -> Tuple[str, List[Dict], Dict]:
        """
        自动重检索（调整参数）
        """
        logger.info(f"🔄 执行重检索，调整检索参数...")

        # 调整检索参数（扩大范围）
        adjusted_params = decision.retrieval_params.copy()
        adjusted_params["top_k"] = min(
            10, adjusted_params["top_k"] * 2
        )  # top_k翻倍（最大10）
        adjusted_params["threshold"] = max(
            0.4, adjusted_params["threshold"] * 0.8
        )  # 阈值降低20%（最低0.4）

        # 执行重检索
        context_str, results, metadata = self.retriever.retrieve(
            query=decision.query_rewritten or query, top_k=adjusted_params["top_k"]
        )

        # 更新元数据
        metadata["reretrieval"] = True
        metadata["adjusted_params"] = adjusted_params

        logger.info(f"✅ 重检索完成，新参数: {adjusted_params}, 结果数: {len(results)}")

        return context_str, results, metadata

    async def query(
        self,
        query: str,
        history: List[Dict] = None,
        stream: bool = False,
        enable_reretrieval: bool = True,
    ) -> Dict[str, any]:
        """
        Agentic RAG完整查询流程

        Returns:
            {
                "answer": str,               # 生成的回答
                "sources": List[Dict],       # 检索到的来源
                "context": str,              # 检索到的上下文
                "metadata": Dict[str, any],  # 元数据
                "generation_metrics": Dict,  # 生成相关指标
                "agent_decision": Dict,      # Agent决策结果
                "quality_check": Dict        # 质量校验结果
            }
        """
        start_time = time.time()
        final_result = {
            "query": query,
            "answer": "",
            "sources": [],
            "context": "",
            "metadata": {},
            "generation_metrics": {},
            "agent_decision": {},
            "quality_check": {},
        }

        try:
            # ========== Step 1: Agent意图分析与决策 ==========
            logger.info(f"🧠 Agent分析查询意图: {query}")
            decision: AgentDecision = await self.llm_service.analyze_query_intent(
                query, history
            )
            final_result["agent_decision"] = decision.model_dump()

            # 直接回答（无需检索）
            if decision.decision_type == "direct_answer":
                logger.info(f"💡 Agent决策：直接回答，无需检索")
                # 直接生成回答
                generation_result = await self.llm_service.generate_response(
                    query=query,
                    context="无需检索的通用政务问题",
                    history=history,
                    decision=decision,
                    stream=stream,
                )

                final_result.update(
                    {
                        "answer": generation_result["answer"],
                        "generation_metrics": generation_result,
                        "quality_check": generation_result.get("quality_check", {}),
                    }
                )
                return final_result

            # 无法回答
            if decision.decision_type == "cannot_answer":
                logger.info(f"🚫 Agent决策：无法回答该问题")
                final_result["answer"] = (
                    "抱歉，我无法回答该问题。请确认问题是否属于泸州市政务范畴，或通过政务服务热线12345咨询。"
                )
                final_result["quality_check"] = {"overall_score": 0.0}
                return final_result

            # ========== Step 2: 基于Agent决策的检索 ==========
            logger.info(
                f"🔍 Agent驱动检索，策略: {decision.retrieval_strategy}, 参数: {decision.retrieval_params}"
            )

            # 执行检索
            context_str, results, metadata = self.retriever.retrieve(
                query=decision.query_rewritten or query,
                top_k=decision.retrieval_params.get("top_k", 5),
            )

            retrieval_time = time.time() - start_time
            metadata["retrieval_time"] = retrieval_time
            logger.info(
                f"✅ 初始检索完成，耗时: {retrieval_time:.2f}s，找到 {len(results)} 个结果"
            )

            # ========== Step 3: 检索结果质量评估 ==========
            retrieval_quality = await self._evaluate_retrieval_quality(
                query, results, metadata
            )
            metadata["retrieval_quality"] = retrieval_quality
            logger.info(
                f"📊 检索质量评分: {retrieval_quality['retrieval_quality_score']:.3f}"
            )

            # 自动重检索（如果开启且需要）
            if enable_reretrieval and retrieval_quality["need_reretrieval"]:
                context_str, results, metadata = (
                    await self._reretrieve_with_adjusted_params(query, decision)
                )
                # 重新评估重检索结果
                retrieval_quality = await self._evaluate_retrieval_quality(
                    query, results, metadata
                )

            # ========== Step 4: 智能生成回答 ==========
            logger.info(f"🤖 Agent开始生成回答...")
            generation_start = time.time()

            if stream:
                # 流式生成（返回生成器）
                final_result["stream_generator"] = (
                    await self.llm_service.generate_response(
                        query=query,
                        context=context_str,
                        history=history,
                        decision=decision,
                        stream=stream,
                    )
                )
            else:
                # 普通生成
                generation_result = await self.llm_service.generate_response(
                    query=query, context=context_str, history=history, decision=decision
                )

                generation_time = time.time() - generation_start
                metadata["generation_time"] = generation_time

                # ========== Step 5: 整合最终结果 ==========
                total_time = time.time() - start_time

                final_result.update(
                    {
                        "answer": generation_result["answer"],
                        "sources": results[: decision.retrieval_params.get("top_k", 5)],
                        "context": context_str,
                        "metadata": {
                            **metadata,
                            "total_time": total_time,
                            "model": generation_result.get("model", "unknown"),
                            "token_usage": generation_result.get("usage", {}),
                        },
                        "generation_metrics": generation_result,
                        "quality_check": generation_result.get("quality_check", {}),
                    }
                )

            return final_result

        except Exception as e:
            logger.error(f"❌ Agentic RAG流程失败: {e}")
            final_result["answer"] = f"抱歉，处理您的查询时出现错误：{str(e)}"
            final_result["metadata"]["error"] = str(e)
            return final_result


# 工具函数
async def query_agentic_rag(
    query: str, history: List[Dict] = None, top_k: int = 5
) -> Dict[str, any]:
    """快速调用Agentic RAG查询"""
    engine = AgenticRAGEngine()
    return await engine.query(query, history)


# 兼容旧接口
async def query_rag(query: str, top_k: int = 5) -> Dict[str, any]:
    """兼容传统RAG接口"""
    engine = AgenticRAGEngine()
    return await engine.query(query, top_k=top_k)


"""
Agentic RAG协调引擎 - 核心决策与流程控制
增强能力：Agent决策驱动、多策略检索、结果评估、自动重检索
"""

import time
from typing import Dict, List, Optional, Tuple
from app.services.retriever import HybridVectorRetriever
from app.services.llm_service import LLMService, AgentDecision, get_llm_service
from app.core.logger import get_logger

logger = get_logger(__name__)


class AgenticRAGEngine:
    """
    Agentic RAG协调引擎
    核心流程：
    1. Agent意图分析与决策
    2. 基于决策的多策略检索
    3. 检索结果质量评估
    4. 智能生成与质量校验
    5. 自动重检索（可选）
    """

    def __init__(self):
        self.retriever = HybridVectorRetriever()
        self.llm_service: LLMService = get_llm_service()
        self.min_quality_score = 0.7  # 最低回答质量阈值
        logger.info("🤖 Agentic RAG引擎初始化完成")

    async def _evaluate_retrieval_quality(
        self, query: str, results: List[Dict], metadata: Dict
    ) -> Dict[str, any]:
        """
        Agent核心能力：评估检索结果质量
        检查：相关性、覆盖度、权威性
        """
        if not results:
            return {
                "retrieval_quality_score": 0.0,
                "suggestion": "无检索结果，建议扩大检索范围",
                "need_reretrieval": True,
            }

        # 计算核心指标
        avg_similarity = metadata.get("avg_similarity", 0.0)
        num_results = metadata.get("num_results", 0)
        dept_coverage = len(set([r["entity"].get("department", "") for r in results]))

        # 综合检索质量评分
        retrieval_score = (
            avg_similarity * 0.6  # 相似度权重60%
            + min(1.0, num_results / 10) * 0.2  # 数量权重20%
            + min(1.0, dept_coverage / 5) * 0.2  # 部门覆盖度权重20%
        )

        # 判断是否需要重检索
        need_reretrieval = retrieval_score < 0.6 or (  # 低质量检索结果
            num_results < 2 and avg_similarity < 0.7
        )  # 结果少且相似度低

        return {
            "retrieval_quality_score": retrieval_score,
            "avg_similarity": avg_similarity,
            "num_results": num_results,
            "dept_coverage": dept_coverage,
            "suggestion": "需要重检索" if need_reretrieval else "检索结果合格",
            "need_reretrieval": need_reretrieval,
        }

    async def _reretrieve_with_adjusted_params(
        self, query: str, decision: AgentDecision
    ) -> Tuple[str, List[Dict], Dict]:
        """
        自动重检索（调整参数）
        """
        logger.info(f"🔄 执行重检索，调整检索参数...")

        # 调整检索参数（扩大范围）
        adjusted_params = decision.retrieval_params.copy()
        adjusted_params["top_k"] = min(
            10, adjusted_params["top_k"] * 2
        )  # top_k翻倍（最大10）
        adjusted_params["threshold"] = max(
            0.4, adjusted_params["threshold"] * 0.8
        )  # 阈值降低20%（最低0.4）

        # 执行重检索
        context_str, results, metadata = self.retriever.retrieve(
            query=decision.query_rewritten or query, top_k=adjusted_params["top_k"]
        )

        # 更新元数据
        metadata["reretrieval"] = True
        metadata["adjusted_params"] = adjusted_params

        logger.info(f"✅ 重检索完成，新参数: {adjusted_params}, 结果数: {len(results)}")

        return context_str, results, metadata

    async def query(
        self,
        query: str,
        history: List[Dict] = None,
        stream: bool = False,
        enable_reretrieval: bool = True,
    ) -> Dict[str, any]:
        """
        Agentic RAG完整查询流程

        Returns:
            {
                "answer": str,               # 生成的回答
                "sources": List[Dict],       # 检索到的来源
                "context": str,              # 检索到的上下文
                "metadata": Dict[str, any],  # 元数据
                "generation_metrics": Dict,  # 生成相关指标
                "agent_decision": Dict,      # Agent决策结果
                "quality_check": Dict        # 质量校验结果
            }
        """
        start_time = time.time()
        final_result = {
            "query": query,
            "answer": "",
            "sources": [],
            "context": "",
            "metadata": {},
            "generation_metrics": {},
            "agent_decision": {},
            "quality_check": {},
        }

        try:
            # ========== Step 1: Agent意图分析与决策 ==========
            logger.info(f"🧠 Agent分析查询意图: {query}")
            decision: AgentDecision = await self.llm_service.analyze_query_intent(
                query, history
            )
            final_result["agent_decision"] = decision.model_dump()

            # 直接回答（无需检索）
            if decision.decision_type == "direct_answer":
                logger.info(f"💡 Agent决策：直接回答，无需检索")
                # 直接生成回答
                generation_result = await self.llm_service.generate_response(
                    query=query,
                    context="无需检索的通用政务问题",
                    history=history,
                    decision=decision,
                    stream=stream,
                )

                final_result.update(
                    {
                        "answer": generation_result["answer"],
                        "generation_metrics": generation_result,
                        "quality_check": generation_result.get("quality_check", {}),
                    }
                )
                return final_result

            # 无法回答
            if decision.decision_type == "cannot_answer":
                logger.info(f"🚫 Agent决策：无法回答该问题")
                final_result["answer"] = (
                    "抱歉，我无法回答该问题。请确认问题是否属于泸州市政务范畴，或通过政务服务热线12345咨询。"
                )
                final_result["quality_check"] = {"overall_score": 0.0}
                return final_result

            # ========== Step 2: 基于Agent决策的检索 ==========
            logger.info(
                f"🔍 Agent驱动检索，策略: {decision.retrieval_strategy}, 参数: {decision.retrieval_params}"
            )

            # 执行检索
            context_str, results, metadata = self.retriever.retrieve(
                query=decision.query_rewritten or query,
                top_k=decision.retrieval_params.get("top_k", 5),
            )

            retrieval_time = time.time() - start_time
            metadata["retrieval_time"] = retrieval_time
            logger.info(
                f"✅ 初始检索完成，耗时: {retrieval_time:.2f}s，找到 {len(results)} 个结果"
            )

            # ========== Step 3: 检索结果质量评估 ==========
            retrieval_quality = await self._evaluate_retrieval_quality(
                query, results, metadata
            )
            metadata["retrieval_quality"] = retrieval_quality
            logger.info(
                f"📊 检索质量评分: {retrieval_quality['retrieval_quality_score']:.3f}"
            )

            # 自动重检索（如果开启且需要）
            if enable_reretrieval and retrieval_quality["need_reretrieval"]:
                context_str, results, metadata = (
                    await self._reretrieve_with_adjusted_params(query, decision)
                )
                # 重新评估重检索结果
                retrieval_quality = await self._evaluate_retrieval_quality(
                    query, results, metadata
                )

            # ========== Step 4: 智能生成回答 ==========
            logger.info(f"🤖 Agent开始生成回答...")
            generation_start = time.time()

            if stream:
                # 流式生成（返回生成器）
                final_result["stream_generator"] = (
                    await self.llm_service.generate_response(
                        query=query,
                        context=context_str,
                        history=history,
                        decision=decision,
                        stream=stream,
                    )
                )
            else:
                # 普通生成
                generation_result = await self.llm_service.generate_response(
                    query=query, context=context_str, history=history, decision=decision
                )

                generation_time = time.time() - generation_start
                metadata["generation_time"] = generation_time

                # ========== Step 5: 整合最终结果 ==========
                total_time = time.time() - start_time

                final_result.update(
                    {
                        "answer": generation_result["answer"],
                        "sources": results[: decision.retrieval_params.get("top_k", 5)],
                        "context": context_str,
                        "metadata": {
                            **metadata,
                            "total_time": total_time,
                            "model": generation_result.get("model", "unknown"),
                            "token_usage": generation_result.get("usage", {}),
                        },
                        "generation_metrics": generation_result,
                        "quality_check": generation_result.get("quality_check", {}),
                    }
                )

            return final_result

        except Exception as e:
            logger.error(f"❌ Agentic RAG流程失败: {e}")
            final_result["answer"] = f"抱歉，处理您的查询时出现错误：{str(e)}"
            final_result["metadata"]["error"] = str(e)
            return final_result


# 工具函数
async def query_agentic_rag(
    query: str, history: List[Dict] = None, top_k: int = 5
) -> Dict[str, any]:
    """快速调用Agentic RAG查询"""
    engine = AgenticRAGEngine()
    return await engine.query(query, history)


# 兼容旧接口
async def query_rag(query: str, top_k: int = 5) -> Dict[str, any]:
    """兼容传统RAG接口"""
    engine = AgenticRAGEngine()
    return await engine.query(query, top_k=top_k)
