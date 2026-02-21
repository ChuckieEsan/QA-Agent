"""
LLM 生成器
基于多模型架构，使用主模型（heavy model）生成回答

架构说明：
- 使用主模型（qwen-max）生成复杂回答
- 轻量任务由分类器等组件单独处理
"""

import json
import traceback
from typing import Dict, List, Optional, AsyncGenerator
from src.app.components.generators.base_generator import BaseGenerator
from src.app.infra.llm.multi_model_service import (
    get_heavy_llm_service,
    get_light_llm_service,
    LLMService,
)
from src.app.infra.utils.logger import get_logger
from dashscope import Generation

logger = get_logger(__name__)


class LLMGenerator(BaseGenerator):
    """
    LLM 生成器

    使用主模型（heavy model）生成高质量回答
    """

    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(LLMGenerator, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        # 使用主模型生成回答
        self.llm_service: LLMService = get_heavy_llm_service()

        self._initialized = True
        logger.info(
            f"✅ LLM Generator 初始化完成（使用主模型: {self.llm_service.get_model_name()}）"
        )

    async def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        history: Optional[List[Dict]] = None,
        **kwargs,
    ) -> str:
        """
        同步生成文本（使用主模型）

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

        # 使用主模型生成回答
        try:
            response = Generation.call(
                model=self.llm_service.get_model_name(),
                prompt=full_prompt,
                temperature=self.llm_service.get_config().temperature,
                max_tokens=self.llm_service.get_config().max_tokens,
                top_p=self.llm_service.get_config().top_p,
                result_format="text",  # 注意：使用 text 格式
            )

            if response.status_code == 200:
                # 注意：result_format='text' 时，结果在 output.text 而不是 choices
                return response.output.text
            else:
                raise Exception(f"API调用失败: {response.code} - {response.message}")

        except Exception as e:
            logger.error(f"❌ LLM 生成失败: {e}")
            logger.error(traceback.format_exc())
            return "抱歉，生成回答时出现错误，请稍后重试。"

    async def generate_stream(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        history: Optional[List[Dict]] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        流式生成文本（使用主模型）

        Args:
            prompt: 用户提示词
            system_message: 系统消息（可选）
            history: 对话历史（可选）
            **kwargs: 其他生成参数

        Yields:
            生成的文本片段
        """
        full_prompt = self._build_prompt(prompt, system_message, history)

        # 使用主模型流式生成
        try:
            response = Generation.call(
                model=self.llm_service.get_model_name(),
                prompt=full_prompt,
                temperature=self.llm_service.get_config().temperature,
                max_tokens=self.llm_service.get_config().max_tokens,
                top_p=self.llm_service.get_config().top_p,
                stream=True,
                result_format="text",  # 注意：流式也使用 text 格式
            )

            for chunk in response:
                if chunk.status_code == 200:
                    # 注意：流式响应中，text 在 output.text 中
                    if chunk.output and hasattr(chunk.output, "text"):
                        text = chunk.output.text
                        if text:
                            yield text
                    # 或者检查 choices（某些版本可能有）
                    elif hasattr(chunk.output, "choices") and chunk.output.choices:
                        for choice in chunk.output.choices:
                            if hasattr(choice, "message") and choice.message.content:
                                yield choice.message.content
                else:
                    yield f"错误: {chunk.code} - {chunk.message}"

        except Exception as e:
            logger.error(f"❌ LLM 流式生成失败: {e}")
            logger.error(traceback.format_exc())
            yield "抱歉，生成回答时出现错误，请稍后重试。"

    async def generate_with_validation(
        self, prompt: str, validation_criteria: Dict[str, any], **kwargs
    ) -> Dict[str, any]:
        """
        生成并验证文本（使用主模型）

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

        # 质量校验（使用轻量模型降低成本）
        quality_score = await self._validate_with_light_model(text, prompt)

        return {
            "text": text,
            "quality_score": quality_score,
            "passed_validation": quality_score > 0.7,
            "validation_details": {"score": quality_score},
        }

    async def _validate_with_light_model(self, text: str, query: str) -> float:
        """
        使用轻量模型进行质量校验（降低成本）
        """
        try:
            validation_prompt = f"""
            请评估以下回答的质量（1-10分）：

            问题：{query}
            回答：{text}

            评分标准：
            1. 相关性：是否直接回答问题
            2. 完整性：信息是否完整
            3. 准确性：是否准确
            4. 清晰度：表达是否清晰

            输出格式（纯数字）：{{
                "score": 0.0-10.0
            }}
            """

            light_llm = get_light_llm_service()
            response = Generation.call(
                model=light_llm.get_model_name(),
                prompt=validation_prompt,
                temperature=light_llm.get_config().temperature,
                max_tokens=100,
                result_format="text",
            )

            if response.status_code == 200:
                # 注意：result_format='text' 时，结果在 output.text 而不是 choices
                result_str = response.output.text
                result = json.loads(result_str)
                score = result.get("score", 5.0)
                return min(score / 10.0, 1.0)  # 转换为 0-1 范围
            else:
                return 0.5  # 默认分数

        except Exception as e:
            logger.warning(f"⚠️  质量校验失败: {e}")
            logger.warning(traceback.format_exc())
            return 0.5  # 默认分数

    async def initialize(self) -> None:
        """初始化生成器资源"""
        # 预热主模型
        await self.generate("Hello")
        logger.info("✅ LLM Generator 预热完成")

    def _build_prompt(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        history: Optional[List[Dict]] = None,
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
