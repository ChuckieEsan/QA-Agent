"""
分类工具
封装 GovClassifier 组件
"""

import traceback
from typing import Dict, Any, Optional
from src.app.agents.tools.base_tool import BaseTool
from src.app.agents.tools.registry import ToolRegistry
from src.app.components.classifier import BaseClassifier, GovClassifier
from src.app.infra.utils.logger import get_logger

logger = get_logger(__name__)


@ToolRegistry.register()
class ClassificationTool(BaseTool):
    """
    分类工具

    封装 GovClassifier，提供问政类型分类能力
    """

    name = "classify"
    description = "分类问政类型（建议/投诉/求助/咨询）"

    def __init__(self, classifier: Optional[BaseClassifier] = None):
        """
        初始化分类工具

        Args:
            classifier: 分类器实例（可选，如果未提供则创建默认实例）
        """
        self.classifier = classifier or GovClassifier()

    async def execute(
        self,
        query: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        执行问政分类

        Args:
            query: 用户查询
            **kwargs: 其他参数

        Returns:
            {
                "type": str,           # 问政类型（建议/投诉/求助/咨询）
                "confidence": float,   # 置信度（0-1）
                "reason": str          # 判定理由（可选）
            }
        """
        try:
            logger.debug(f"📋 [ClassificationTool] 分类: {query[:50]}...")

            # 执行分类
            classification = await self.classifier.classify_gov_request(query)

            logger.debug(
                f"  → 分类结果: {classification['type']} "
                f"(置信度: {classification['confidence']:.2f})"
            )

            return classification

        except Exception as e:
            logger.error(f"❌ [ClassificationTool] 分类失败: {e}")
            logger.error(traceback.format_exc())
            return {
                "type": "未知",
                "confidence": 0.0,
                "reason": f"分类失败: {str(e)}"
            }

    def get_schema(self) -> Dict[str, Any]:
        """获取工具 Schema"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "query": "用户查询语句"
            }
        }

