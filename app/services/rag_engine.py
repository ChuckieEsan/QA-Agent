"""
RAG协调引擎 - 整合检索和生成
"""

import time
from typing import Dict, List, Optional, Tuple
from app.services.retriever import HybridVectorRetriever
from app.services.llm_service import LLMService
from app.core.logger import get_logger

logger = get_logger(__name__)


class RAGEngine:
    """
    RAG协调引擎
    职责：调用检索器 -> 调用LLM生成 -> 返回完整结果
    """
    
    def __init__(self):
        self.retriever = HybridVectorRetriever()
        self.llm_service = LLMService()
        logger.info("🔄 RAG引擎初始化完成")
    
    async def query(
        self, 
        query: str, 
        top_k: int = 5,
        history: List[Dict] = None,
        stream: bool = False
    ) -> Dict[str, any]:
        """
        完整的RAG查询流程
        
        Returns:
            {
                "answer": str,               # 生成的回答
                "sources": List[Dict],       # 检索到的来源
                "context": str,              # 检索到的上下文
                "metadata": Dict[str, any],  # 元数据
                "generation_metrics": Dict   # 生成相关指标
            }
        """
        start_time = time.time()
        
        # 1. 检索阶段
        logger.info(f"🔍 检索查询: {query}")
        context_str, results, metadata = self.retriever.retrieve(query, top_k)
        
        retrieval_time = time.time() - start_time
        logger.info(f"✅ 检索完成，耗时: {retrieval_time:.2f}s，找到 {len(results)} 个结果")
        
        # 2. 生成阶段
        logger.info(f"🤖 开始生成回答...")
        generation_start = time.time()
        
        if stream:
            # 流式生成
            return await self._stream_generation(query, context_str, results, metadata, history)
        else:
            # 普通生成
            generation_result = await self.llm_service.generate_response(
                query=query,
                context=context_str,
                history=history
            )
        
        generation_time = time.time() - generation_start
        
        # 3. 整合结果
        total_time = time.time() - start_time
        
        return {
            "query": query,
            "answer": generation_result["answer"],
            "sources": results[:top_k],
            "context": context_str,
            "metadata": {
                **metadata,
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": total_time,
                "model": generation_result.get("model", "unknown"),
                "token_usage": generation_result.get("usage", {})
            },
            "generation_metrics": generation_result
        }
    
    async def _stream_generation(self, query, context, results, metadata, history):
        """流式生成处理"""
        # 实现流式生成逻辑
        raise NotImplementedError("流式生成尚未实现")


# 工具函数
async def query_rag(query: str, top_k: int = 5) -> Dict[str, any]:
    """快速RAG查询"""
    engine = RAGEngine()
    return await engine.query(query, top_k)