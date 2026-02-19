"""
LLM生成服务 - Agentic RAG 增强版
负责：意图分析、Prompt优化、回答校验、决策生成
"""

import asyncio
import json
from typing import Dict, List, Optional, AsyncGenerator, Literal
from datetime import datetime

import dashscope
from dashscope import Generation
from src.config.setting import settings
from src.app.infra.utils.logger import get_logger
from src.app.infra.llm.schema import Message, ContentItem, SYSTEM, USER, ASSISTANT, BaseModelCompatibleDict
from src.app.infra.llm.base_llm_service import BaseLLMService

logger = get_logger(__name__)

# Agent 决策类型
AgentDecisionType = Literal[
    "direct_answer",    # 无需检索，直接回答
    "need_retrieval",   # 需要检索后回答
    "multi_retrieval",  # 需要多策略检索
    "cannot_answer"     # 无法回答
]

# 检索策略类型
RetrievalStrategy = Literal[
    "hybrid",           # 混合向量检索（默认）
    "keyword",          # 关键词检索
    "semantic_only",    # 纯语义检索
    "cross_dept"        # 跨部门检索
]

class AgentDecision(BaseModelCompatibleDict):
    """Agent 决策结果模型"""
    decision_type: AgentDecisionType
    retrieval_strategy: Optional[RetrievalStrategy] = None
    retrieval_params: Optional[Dict] = None  # top_k/threshold 等参数
    query_rewritten: Optional[str] = None    # 重写后的查询
    intent: str = ""                         # 查询意图
    confidence: float = 0.0                  # 决策置信度

class LLMService(BaseLLMService):
    """
    LLM生成服务（Agentic RAG 增强版）
    新增能力：意图分析、检索决策、Prompt优化、回答校验
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(LLMService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_is_initialized", False):
            return

        logger.info("🔄 初始化Agentic LLM生成服务...")

        # 配置API密钥
        dashscope.api_key = settings.llm.api_key

        # 模型配置
        self.model_name = settings.llm.model_name
        self.temperature = settings.llm.temperature
        self.max_tokens = settings.llm.max_tokens
        self.top_p = settings.llm.top_p

        # 系统Prompt（Agentic RAG 增强版）
        self.system_prompt = self._build_system_prompt()
        self.agent_decision_prompt = self._build_agent_decision_prompt()

        # 缓存最近对话历史
        self.conversation_cache = {}

        self._is_initialized = True
        logger.info(f"✅ Agentic LLM服务初始化完成，使用模型: {self.model_name}")

    def _build_system_prompt(self) -> str:
        """构建Agentic RAG 专用系统提示词"""
        return """你是一名具备智能决策能力的政务问答Agent，专门回答泸州市相关的政策咨询和民生问题。

# 核心能力：
1. **意图理解**：精准识别用户查询的核心意图和信息需求
2. **检索增强**：基于检索到的案例信息，准确、完整地回答问题
3. **来源溯源**：必须引用案例中的具体信息，并注明来源部门和时间
4. **质量校验**：确保回答准确、合规、符合政务沟通规范
5. **自我修正**：如果检索信息不足，明确告知用户并提供替代咨询途径

# 回答规范：
1. 结构化输出：复杂问题分点说明，关键信息加粗
2. 来源标注：格式为【来源：XX部门 | 时间：YYYY-MM-DD】
3. 时效性说明：注明政策的有效时间范围
4. 兜底说明：信息不足时，提供相关部门联系方式

# 禁止行为：
- 不编造未在案例中出现的信息
- 不泄露个人隐私或敏感信息
- 不回答与泸州市无关的问题"""

    def _build_agent_decision_prompt(self) -> str:
        """构建Agent决策专用Prompt（检索前意图分析）"""
        return """你是一名RAG Agent决策助手，负责分析用户查询并给出检索决策。

# 决策任务：
1. 分析用户查询的核心意图
2. 判断是否需要检索知识库
3. 选择最优检索策略
4. 调整检索参数（如top_k、阈值）
5. 必要时重写查询语句

# 决策规则：
- direct_answer：通用政务常识、无需具体案例支撑的问题（如"如何办理身份证"的通用流程）
- need_retrieval：需要具体政策/案例支撑的问题（如"2024年泸州雨露计划补贴标准"）
- multi_retrieval：跨部门/多政策的复杂问题（如"泸州小微企业税收优惠+社保补贴"）
- cannot_answer：非泸州市政务问题/无意义问题/敏感问题

# 检索策略选择：
- hybrid：默认策略，混合语义+关键词检索
- keyword：强关键词特征的问题（如"2024年泸州医保缴费标准"）
- semantic_only：语义模糊/多义词问题（如"泸州创业扶持政策"）
- cross_dept：跨部门问题（如"泸州住房补贴+公积金政策"）

# 输出格式（JSON）：
{
    "decision_type": "direct_answer|need_retrieval|multi_retrieval|cannot_answer",
    "retrieval_strategy": "hybrid|keyword|semantic_only|cross_dept",
    "retrieval_params": {"top_k": 5-10, "threshold": 0.5-0.8},
    "query_rewritten": "重写后的查询语句（可选）",
    "intent": "核心意图描述",
    "confidence": 0.0-1.0
}

# 注意：
- retrieval_strategy/cross_dept 仅在decision_type为need_retrieval/multi_retrieval时必填
- retrieval_params需根据问题复杂度调整（复杂问题top_k=8-10，简单问题=3-5）
- query_rewritten需更精准表达核心意图（如原问题"雨露计划多少钱"→"2024年泸州市雨露计划补贴金额标准"）"""

    async def analyze_query_intent(self, query: str, history: List[Dict] = None) -> AgentDecision:
        """
        Agent核心能力：分析查询意图并生成检索决策
        """
        # 构建决策Prompt
        prompt_parts = [
            self.agent_decision_prompt,
            "\n# 用户查询：",
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
            # 调用LLM生成决策
            response = Generation.call(
                model=self.model_name,
                prompt=prompt,
                temperature=0.1,  # 决策阶段低随机性
                max_tokens=500,
                top_p=0.9,
                result_format='text'
            )

            if response.status_code == 200:
                decision_str = response.output.choices[0].message.content
                # 解析JSON决策
                decision_data = json.loads(decision_str)
                return AgentDecision(**decision_data)
            else:
                raise Exception(f"决策生成失败: {response.code} - {response.message}")

        except Exception as e:
            logger.error(f"❌ Agent决策失败: {e}")
            # 返回默认决策（兜底）
            return AgentDecision(
                decision_type="need_retrieval",
                retrieval_strategy="hybrid",
                retrieval_params={"top_k": 5, "threshold": 0.6},
                query_rewritten=query,
                intent=f"无法解析意图：{str(e)}",
                confidence=0.5
            )

    def build_agent_rag_prompt(self, query: str, context: str, decision: AgentDecision, history: List[Dict] = None) -> str:
        """
        构建Agentic RAG专用Prompt（优化版）
        结合Agent决策结果，动态优化Prompt
        """
        prompt_parts = []

        # 1. 系统指令（增强版）
        prompt_parts.append(f"系统指令：{self.system_prompt}")
        prompt_parts.append("")

        # 2. Agent决策信息
        prompt_parts.append(f"## Agent决策信息")
        prompt_parts.append(f"查询意图：{decision.intent}")
        prompt_parts.append(f"检索策略：{decision.retrieval_strategy}")
        prompt_parts.append("")

        # 3. 对话历史
        if history and len(history) > 0:
            prompt_parts.append("## 对话历史")
            for i, turn in enumerate(history[-3:]):
                role = "用户" if turn["role"] == "user" else "助手"
                prompt_parts.append(f"{role}：{turn['content']}")
            prompt_parts.append("")

        # 4. 检索上下文（增强标注）
        prompt_parts.append("## 检索到的权威案例信息")
        prompt_parts.append(context)
        prompt_parts.append("")

        # 5. 优化后的查询
        prompt_parts.append("## 用户核心问题")
        prompt_parts.append(decision.query_rewritten or query)
        prompt_parts.append("")

        # 6. 动态回答要求（基于决策类型）
        prompt_parts.append("## 回答要求")
        if decision.decision_type == "multi_retrieval":
            prompt_parts.append("1. 分部门/分政策维度回答")
            prompt_parts.append("2. 明确各维度信息的来源和时间")
            prompt_parts.append("3. 总结各维度信息的关联性")
        else:
            prompt_parts.append("1. 精准引用案例中的具体数据和政策条款")
            prompt_parts.append("2. 按【来源：XX部门 | 时间：YYYY-MM-DD】格式标注来源")
            prompt_parts.append("3. 语言简洁、专业，符合政务沟通规范")

        return "\n".join(prompt_parts)

    async def validate_answer_quality(self, answer: str, query: str, context: str) -> Dict[str, any]:
        """
        Agent核心能力：回答质量校验
        检查：相关性、准确性、来源标注、合规性
        """
        validate_prompt = f"""
        你是回答质量校验Agent，请校验以下回答是否符合要求：

        ## 校验标准
        1. 相关性：回答是否与用户查询({query})直接相关
        2. 准确性：是否仅基于提供的上下文信息，无编造内容
        3. 来源标注：是否注明信息来源部门和时间
        4. 合规性：是否符合政务沟通规范，无敏感信息

        ## 待校验内容
        上下文：{context[:1000]}...
        回答：{answer}

        ## 输出格式（JSON）
        {{
            "relevance_score": 0.0-1.0,
            "accuracy_score": 0.0-1.0,
            "attribution_score": 0.0-1.0,
            "compliance_score": 0.0-1.0,
            "overall_score": 0.0-1.0,
            "suggestion": "优化建议（可选）"
        }}
        """

        try:
            response = Generation.call(
                model=self.model_name,
                prompt=validate_prompt,
                temperature=0.1,
                max_tokens=300,
                result_format='text'
            )

            if response.status_code == 200:
                validate_result = json.loads(response.output.choices[0].message.content)
                return validate_result
            else:
                raise Exception(f"质量校验失败: {response.code}")

        except Exception as e:
            logger.error(f"❌ 回答质量校验失败: {e}")
            return {
                "relevance_score": 0.0,
                "accuracy_score": 0.0,
                "attribution_score": 0.0,
                "compliance_score": 0.0,
                "overall_score": 0.0,
                "suggestion": f"校验失败：{str(e)}"
            }

    async def generate_response(
        self,
        query: str,
        context: str,
        history: List[Dict] = None,
        decision: Optional[AgentDecision] = None,
        stream: bool = False
    ) -> Dict[str, any]:
        """
        增强版生成回答（结合Agent决策）
        """
        start_time = datetime.now()

        try:
            # 如果没有决策结果，先执行意图分析
            if not decision:
                decision = await self.analyze_query_intent(query, history)

            # 构建Agent优化后的Prompt
            prompt = self.build_agent_rag_prompt(query, context, decision, history)

            logger.debug(f"Agentic Prompt长度: {len(prompt)}字符")
            logger.info(f"Agent决策类型: {decision.decision_type}, 检索策略: {decision.retrieval_strategy}")

            # 调用LLM生成回答
            if stream:
                return await self._generate_stream(prompt)
            else:
                generation_result = await self._generate_once(prompt, start_time)

                # 回答质量校验
                quality_check = await self.validate_answer_quality(
                    generation_result["answer"], query, context
                )

                # 整合校验结果
                generation_result["quality_check"] = quality_check
                generation_result["agent_decision"] = decision.model_dump()

                return generation_result

        except Exception as e:
            logger.error(f"❌ Agentic LLM生成失败: {e}")
            return {
                "answer": "抱歉，生成回答时出现错误，请稍后重试。",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "quality_check": {"overall_score": 0.0},
                "agent_decision": {"decision_type": "cannot_answer"}
            }

    async def _generate_once(self, prompt: str, start_time: datetime) -> Dict[str, any]:
        """一次性生成完整回答"""
        response = Generation.call(
            model=self.model_name,
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            result_format='message'
        )

        if response.status_code == 200:
            answer = response.output.choices[0].message.content

            return {
                "answer": answer,
                "usage": {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "model": self.model_name,
                "finish_reason": response.output.choices[0].finish_reason,
                "response_time": (datetime.now() - start_time).total_seconds(),
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise Exception(f"API调用失败: {response.code} - {response.message}")

    async def _generate_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """流式生成回答"""
        response = Generation.call(
            model=self.model_name,
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            stream=True,
            result_format='message'
        )

        for chunk in response:
            if chunk.status_code == 200:
                if hasattr(chunk.output, 'choices') and chunk.output.choices:
                    content = chunk.output.choices[0].message.content
                    if content:
                        yield content
            else:
                yield f"错误: {chunk.code} - {chunk.message}"

    async def initialize(self) -> None:
        """初始化 LLM 服务资源"""
        # 预热：执行一次简单的生成
        await self.generate_response(
            query="",
            context="Hello",
            history=None,
            stream=False
        )
        logger.info("✅ LLM Service 预热完成")


# 工具函数
def get_llm_service() -> LLMService:
    """获取Agentic LLM服务单例实例"""
    return LLMService()

async def generate_agentic_rag_response(
    query: str,
    context: str,
    history: List[Dict] = None,
    decision: Optional[AgentDecision] = None
) -> Dict[str, any]:
    """快速生成Agentic RAG回答"""
    service = get_llm_service()
    return await service.generate_response(query, context, history, decision)
