"""
LLM 生成器
基于多模型架构，使用主模型（heavy model）生成回答

架构说明：
- 使用主模型（qwen-max）生成复杂回答
- 轻量任务由分类器等组件单独处理
- 使用 Message 列表进行交互，支持系统/用户/助手/函数消息
"""

import json
import traceback
from typing import Dict, List, Optional, AsyncGenerator, Union, Any
from src.app.components.generators.base_generator import BaseGenerator
from src.app.infra.llm.multi_model_service import (
    get_heavy_llm_service,
    get_light_llm_service,
    LLMService,
)
from src.app.infra.llm.schema import Message, message_list_to_dict
from src.app.infra.utils.logger import get_logger
from dashscope import Generation

logger = get_logger(__name__)


class LLMGenerator(BaseGenerator):
    """
    LLM 生成器

    使用主模型（heavy model）生成高质量回答
    使用 Message 列表进行交互，支持：
    - system/user/assistant/function 角色
    - 多模态内容
    - function_call 格式
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

        # 默认系统消息
        self.system_message = "你是一位专业的政务问答专家，善于分析用户问题、检索相关政策、生成专业回答。你的职责是：提供准确、权威、符合政策的政务咨询服务。"

        self._initialized = True
        logger.info(
            f"✅ LLM Generator 初始化完成（使用主模型: {self.llm_service.get_model_name()}）"
        )

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        **kwargs,
    ) -> str:
        """
        同步生成文本（使用主模型）

        Args:
            messages: 消息列表，每个消息包含:
                - role: "system" | "user" | "assistant" | "function"
                - content: str | List[ContentItem]
                - name: Optional[str]
                - function_call: Optional[dict]
            **kwargs: 其他生成参数（temperature, max_tokens, top_p 等）

        Returns:
            生成的文本
        """
        # 追加系统消息, 准备消息列表
        prepared_messages = self._prepare_messages(messages)

        # 转换为 prompt 字符串
        prompt = self._convert_to_prompt(prepared_messages)

        # 使用主模型生成回答
        try:
            response = Generation.call(
                model=self.llm_service.get_model_name(),
                prompt=prompt,
                temperature=self._get_param(kwargs, 'temperature', self.llm_service.get_config().temperature),
                max_tokens=self._get_param(kwargs, 'max_tokens', self.llm_service.get_config().max_tokens),
                top_p=self._get_param(kwargs, 'top_p', self.llm_service.get_config().top_p),
                result_format="text",
            )

            if response.status_code == 200:
                response_text = response.output.text
                # 将助手回复添加到消息历史
                prepared_messages.append(Message(role="assistant", content=response_text))
                return response_text
            else:
                raise Exception(f"API调用失败: {response.code} - {response.message}")

        except Exception as e:
            logger.error(f"❌ LLM 生成失败: {e}")
            logger.error(traceback.format_exc())
            return "抱歉，生成回答时出现错误，请稍后重试。"

    async def generate_stream(
        self,
        messages: List[Dict[str, Any]],
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        流式生成文本（使用主模型）

        Args:
            messages: 消息列表
            **kwargs: 其他生成参数

        Yields:
            生成的文本片段
        """
        # 追加系统消息, 准备消息列表
        prepared_messages = self._prepare_messages(messages)

        # 转换为 prompt 字符串
        prompt = self._convert_to_prompt(prepared_messages)

        # 使用主模型流式生成
        try:
            response = Generation.call(
                model=self.llm_service.get_model_name(),
                prompt=prompt,
                temperature=self._get_param(kwargs, 'temperature', self.llm_service.get_config().temperature),
                max_tokens=self._get_param(kwargs, 'max_tokens', self.llm_service.get_config().max_tokens),
                top_p=self._get_param(kwargs, 'top_p', self.llm_service.get_config().top_p),
                stream=True,
                result_format="text",
            )

            full_response = ""
            for chunk in response:
                if chunk.status_code == 200:
                    if chunk.output and hasattr(chunk.output, "text"):
                        text = chunk.output.text
                        if text:
                            full_response += text
                            yield text
                else:
                    yield f"错误: {chunk.code} - {chunk.message}"

            # 将助手回复添加到消息历史
            if full_response:
                prepared_messages.append(Message(role="assistant", content=full_response))

        except Exception as e:
            logger.error(f"❌ LLM 流式生成失败: {e}")
            logger.error(traceback.format_exc())
            yield "抱歉，生成回答时出现错误，请稍后重试。"

    async def generate_with_validation(
        self,
        messages: List[Dict[str, Any]],
        validation_criteria: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        生成并验证文本（使用主模型）

        Args:
            messages: 消息列表
            validation_criteria: 验证标准
            **kwargs: 其他生成参数

        Returns:
            {
                "text": str,
                "quality_score": float,
                "passed_validation": bool,
                "validation_details": Dict
            }
        """
        # 生成文本
        text = await self.generate(messages, **kwargs)

        # 质量校验（使用轻量模型降低成本）
        # 提取用户查询用于验证
        query = self._extract_query_from_messages(messages)
        quality_score = await self._validate_with_light_model(text, query)

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
        await self.generate([{"role": "user", "content": "Hello"}])
        logger.info("✅ LLM Generator 预热完成")

    def _extract_query_from_messages(
        self,
        messages: List[Dict[str, Any]]
    ) -> str:
        """
        从消息列表中提取用户查询

        Args:
            messages: 消息列表

        Returns:
            用户查询字符串
        """
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    # 提取文本内容
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            text_parts.append(item["text"])
                        elif hasattr(item, "text"):
                            text_parts.append(item.text)
                    return " ".join(text_parts)
                return str(content)
        return ""

    def _get_param(
        self,
        kwargs: Dict[str, Any],
        key: str,
        default: Any = None
    ) -> Any:
        """
        从 kwargs 中获取参数，优先使用 kwargs

        Args:
            kwargs: 关键字参数
            key: 参数键
            default: 默认值

        Returns:
            参数值
        """
        return kwargs.get(key, default)
