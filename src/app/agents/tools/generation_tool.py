"""
生成工具
封装 LLMGenerator 组件
"""

import traceback
from typing import Dict, Any, Optional, List
from src.app.agents.tools.base_tool import BaseTool
from src.app.agents.tools.registry import ToolRegistry
from src.app.components.generators import BaseGenerator, LLMGenerator
from src.app.infra.utils.logger import get_logger

logger = get_logger(__name__)


@ToolRegistry.register()
class GenerationTool(BaseTool):
    """
    生成工具

    封装 LLMGenerator，提供文本生成能力
    """

    name = "generate"
    description = "当需要整理、归纳、总结信息时，可以生成文本"

    def __init__(self, generator: Optional[BaseGenerator] = None):
        """
        初始化生成工具

        Args:
            generator: 生成器实例（可选，如果未提供则创建默认实例）
        """
        self.generator = generator or LLMGenerator()

    async def execute(
        self,
        prompt: str,
        context: str = "",
        system_message: Optional[str] = None,
        history: Optional[List[Dict]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        执行文本生成

        Args:
            prompt: 用户提示词
            context: 上下文信息（可选）
            system_message: 系统消息（可选）
            history: 对话历史（可选）
            **kwargs: 其他生成参数

        Returns:
            {
                "answer": str,      # 生成的回答文本
                "metadata": Dict    # 元数据
            }
        """
        try:
            logger.debug(f"🤖 [GenerationTool] 生成文本: {prompt[:50]}...")

            # 构建完整的 prompt
            full_prompt = prompt
            if context:
                full_prompt = f"基于以下上下文回答问题：\n\n{context}\n\n问题：{prompt}"

            # 执行生成
            answer = await self.generator.generate(
                prompt=full_prompt,
                system_message=system_message,
                history=history,
                **kwargs
            )

            logger.debug(f"  → 生成完成: {answer[:50]}...")

            return {
                "answer": answer,
                "metadata": {"length": len(answer)}
            }

        except Exception as e:
            logger.error(f"❌ [GenerationTool] 生成失败: {e}")
            logger.error(traceback.format_exc())
            return {
                "answer": f"抱歉，生成回答时出现错误：{str(e)}",
                "metadata": {"error": str(e)}
            }

    def get_schema(self) -> Dict[str, Any]:
        """获取工具 Schema"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "prompt": "用户提示词",
                "context": "上下文信息（可选）",
                "system_message": "系统消息（可选）",
                "history": "对话历史（可选）"
            }
        }

