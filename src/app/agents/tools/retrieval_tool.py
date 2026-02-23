"""
检索工具
封装 HybridVectorRetriever 组件
"""

import traceback
from typing import Dict, Any, Optional
from src.app.agents.tools.base_tool import BaseTool
from src.app.agents.tools.registry import ToolRegistry
from src.app.components.retrievers import BaseRetriever, HybridVectorRetriever
from src.app.infra.utils.logger import get_logger

logger = get_logger(__name__)


@ToolRegistry.register()
class RetrievalTool(BaseTool):
    """
    检索工具

    封装 HybridVectorRetriever，提供向量检索能力
    """

    name = "retrieve"
    description = "检索知识库已有的网络问政相关案例"

    def __init__(self, retriever: Optional[BaseRetriever] = None):
        """
        初始化检索工具

        Args:
            retriever: 检索器实例（可选，如果未提供则创建默认实例）
        """
        self.retriever = retriever or HybridVectorRetriever()

    async def execute(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        执行检索

        Args:
            query: 查询语句
            top_k: 返回结果数量
            threshold: 相似度阈值
            **kwargs: 其他参数

        Returns:
            {
                "context": str,          # 检索到的上下文文本
                "results": List[Dict],   # 检索结果列表
                "metadata": Dict         # 元数据（检索耗时等）
            }
        """
        try:
            logger.debug(f"🔍 [RetrievalTool] 执行检索: {query[:50]}...")

            # 执行检索
            context, results, metadata = self.retriever.retrieve(
                query=query,
                top_k=top_k,
                threshold=threshold
            )

            # 格式化结果
            formatted_results = []
            for idx, result in enumerate(results[:top_k]):
                formatted_results.append({
                    "rank": idx + 1,
                    "title": result.get("title", "无标题"),
                    "department": result.get("department", "未知部门"),
                    "time": result.get("time", "未知时间"),
                    "content": result.get("content", ""),
                    "similarity": result.get("similarity", 0.0),
                    "composite_score": result.get("composite_score", 0.0)
                })

            logger.debug(f"  → 检索到 {len(formatted_results)} 个结果")

            return {
                "context": context,
                "results": formatted_results,
                "metadata": metadata
            }

        except Exception as e:
            logger.error(f"❌ [RetrievalTool] 检索失败: {e}")
            logger.error(traceback.format_exc())
            return {
                "context": "",
                "results": [],
                "metadata": {"error": str(e)}
            }

    def get_schema(self) -> Dict[str, Any]:
        """获取工具 Schema"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "query": "检索查询语句",
                "top_k": "返回结果数量（默认 5）",
                "threshold": "相似度阈值（默认 0.5）"
            }
        }

