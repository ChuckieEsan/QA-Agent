"""
验证工具
封装 AnswerValidator 组件
"""

import traceback
from typing import Dict, Any, Optional
from src.app.agents.tools.base_tool import BaseTool
from src.app.agents.tools.registry import ToolRegistry
from src.app.components.quality import BaseValidator, AnswerValidator
from src.app.infra.utils.logger import get_logger

logger = get_logger(__name__)


@ToolRegistry.register()
class ValidationTool(BaseTool):
    """
    验证工具

    封装 AnswerValidator，提供回答质量验证能力
    """

    name = "validate"
    description = "验证回答是否符合网络问政场景的规范"

    def __init__(self, validator: Optional[BaseValidator] = None):
        """
        初始化验证工具

        Args:
            validator: 验证器实例（可选，如果未提供则创建默认实例）
        """
        self.validator = validator or AnswerValidator()

    async def execute(
        self,
        answer: str,
        query: str,
        context: str = "",
        **kwargs
    ) -> Dict[str, Any]:
        """
        执行回答质量验证

        Args:
            answer: 生成的回答
            query: 用户查询
            context: 上下文信息（可选）
            **kwargs: 其他参数

        Returns:
            {
                "overall_score": float,       # 综合评分（0-1）
                "relevance_score": float,     # 相关性评分
                "completeness_score": float,  # 完整性评分
                "accuracy_score": float,      # 准确性评分
                "passed": bool,               # 是否通过验证
                "feedback": str               # 反馈信息
            }
        """
        try:
            logger.debug(f"🔍 [ValidationTool] 验证回答质量: {answer[:50]}...")

            # 执行验证
            validation = await self.validator.validate(answer, query, context)

            logger.debug(
                f"  → 验证结果: 综合评分 {validation['overall_score']:.2f} "
                f"(通过: {validation['passed']})"
            )

            return validation

        except Exception as e:
            logger.error(f"❌ [ValidationTool] 验证失败: {e}")
            logger.error(traceback.format_exc())
            return {
                "overall_score": 0.0,
                "relevance_score": 0.0,
                "completeness_score": 0.0,
                "accuracy_score": 0.0,
                "passed": False,
                "feedback": f"验证失败: {str(e)}"
            }

    def get_schema(self) -> Dict[str, Any]:
        """获取工具 Schema"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "answer": "生成的回答文本",
                "query": "用户查询",
                "context": "上下文信息（可选）"
            }
        }

