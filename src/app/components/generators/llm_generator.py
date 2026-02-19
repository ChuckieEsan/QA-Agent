"""
LLM 生成器
基于 LLM Service 封装的生成器实现
"""

from typing import Dict, List, Optional, AsyncGenerator
from src.app.components.generators.base_generator import BaseGenerator
from src.app.infra.llm.llm_service import LLMService, get_llm_service
from src.app.infra.utils.logger import get_logger

logger = get_logger(__name__)


class LLMGenerator(BaseGenerator):
    """
    LLM 生成器

    封装 LLM Service，提供更高层次的生成接口
    """

    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(LLMGenerator, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        self.llm_service: LLMService = get_llm_service()
        self._initialized = True
        logger.info("✅ LLM Generator 初始化完成")

    async def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        history: Optional[List[Dict]] = None,
        **kwargs
    ) -> str:
        """
        同步生成文本

        Args:
            prompt: 用户提示词
            system_message: 系统消息（可选）
            history: 对话历史（可选）
            **kwargs: 其他生成参数

        Returns:
            生成的文本
        """
        # 构建完整 Prompt
        full_prompt = self._build_prompt(prompt, system_message, history)

        # 调用 LLM Service
        result = await self.llm_service.generate_response(
            query="",
            context=full_prompt,
            history=history,
            stream=False
        )

        return result["answer"]

    async def generate_stream(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        history: Optional[List[Dict]] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        流式生成文本

        Args:
            prompt: 用户提示词
            system_message: 系统消息（可选）
            history: 对话历史（可选）
            **kwargs: 其他生成参数

        Yields:
            生成的文本片段
        """
        full_prompt = self._build_prompt(prompt, system_message, history)

        # 调用 LLM Service 的流式生成
        async for chunk in await self.llm_service.generate_response(
            query="",
            context=full_prompt,
            history=history,
            stream=True
        ):
            yield chunk

    async def generate_with_validation(
        self,
        prompt: str,
        validation_criteria: Dict[str, any],
        **kwargs
    ) -> Dict[str, any]:
        """
        生成并验证文本

        Args:
            prompt: 用户提示词
            validation_criteria: 验证标准
            **kwargs: 其他生成参数

        Returns:
            {
                "text": str,                    # 生成的文本
                "quality_score": float,         # 质量分数 (0-1)
                "passed_validation": bool,      # 是否通过验证
                "validation_details": Dict      # 验证详情
            }
        """
        # 生成文本
        text = await self.generate(prompt, **kwargs)

        # 调用 LLM Service 的质量校验
        validation_result = await self.llm_service.validate_answer_quality(
            answer=text,
            query=prompt,
            context=""
        )

        return {
            "text": text,
            "quality_score": validation_result.get("overall_score", 0.0),
            "passed_validation": validation_result.get("overall_score", 0) > 0.7,
            "validation_details": validation_result
        }

    async def initialize(self) -> None:
        """初始化生成器资源"""
        # 预热 LLM Service
        await self.llm_service.generate_response(
            query="",
            context="Hello",
            history=None,
            stream=False
        )
        logger.info("✅ LLM Generator 预热完成")

    def _build_prompt(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        history: Optional[List[Dict]] = None
    ) -> str:
        """
        构建完整 Prompt

        Args:
            prompt: 用户提示词
            system_message: 系统消息（可选）
            history: 对话历史（可选）

        Returns:
            完整的 Prompt 字符串
        """
        parts = []

        if system_message:
            parts.append(f"## 系统指令\n{system_message}\n")

        if history:
            parts.append("## 对话历史")
            for msg in history[-3:]:
                role = "用户" if msg["role"] == "user" else "助手"
                parts.append(f"{role}: {msg['content']}")
            parts.append("")

        parts.append(f"## 当前查询\n{prompt}")

        return "\n".join(parts)
