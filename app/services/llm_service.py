"""
LLM生成服务 - 负责与Qwen API交互，生成最终回答
"""

import asyncio
import json
from typing import Dict, List, Optional, AsyncGenerator
from datetime import datetime

import dashscope
from dashscope import Generation
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class LLMService:
    """
    LLM生成服务（单例模式）
    负责：Prompt构建、API调用、流式响应、错误处理
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(LLMService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if getattr(self, "_is_initialized", False):
            return
        
        logger.info("🔄 初始化LLM生成服务...")
        
        # 配置API密钥
        dashscope.api_key = settings.llm.api_key
        
        # 模型配置
        self.model_name = settings.llm.model_name
        self.temperature = settings.llm.temperature
        self.max_tokens = settings.llm.max_tokens
        self.top_p = settings.llm.top_p
        
        # 系统Prompt（政务领域优化）
        self.system_prompt = self._build_system_prompt()
        
        # 缓存最近对话历史（可选）
        self.conversation_cache = {}
        
        self._is_initialized = True
        logger.info(f"✅ LLM服务初始化完成，使用模型: {self.model_name}")
    
    def _build_system_prompt(self) -> str:
        """构建政务领域专用系统提示词"""
        return """你是一名政务问答助手，专门回答泸州市相关的政策咨询和民生问题。

# 你的角色和能力：
1. **政策专家**：熟悉泸州市各级政府部门职责和业务流程
2. **信息整合者**：基于提供的案例信息，准确、完整地回答用户问题
3. **专业沟通者**：语言正式、准确、友好，符合政务沟通规范

# 回答要求：
1. **准确性第一**：只基于提供的案例信息回答，不编造不存在的信息
2. **清晰标注来源**：回答中注明参考的部门和案例时间
3. **结构化输出**：复杂问题分点说明，关键信息突出
4. **时效性说明**：注明政策或信息的有效时间范围
5. **提供后续指引**：给出相关部门的联系方式或进一步咨询途径

# 注意事项：
- 如果案例信息不足或与问题不相关，如实告知用户
- 涉及个人隐私或敏感信息时，提示用户通过正规渠道咨询
- 不同部门的政策可能不同，注意区分说明

现在开始回答用户问题："""
    
    def build_rag_prompt(self, query: str, context: str, history: List[Dict] = None) -> str:
        """
        构建RAG专用Prompt
        
        Args:
            query: 用户查询
            context: 检索到的上下文
            history: 对话历史（可选）
        
        Returns:
            完整的Prompt字符串
        """
        # 基础Prompt结构
        prompt_parts = []
        
        # 1. 系统提示
        prompt_parts.append(f"系统指令：{self.system_prompt}")
        prompt_parts.append("")  # 空行分隔
        
        # 2. 对话历史（如果有）
        if history and len(history) > 0:
            prompt_parts.append("对话历史：")
            for i, turn in enumerate(history[-3:]):  # 只保留最近3轮
                role = "用户" if turn["role"] == "user" else "助手"
                prompt_parts.append(f"{role}: {turn['content']}")
            prompt_parts.append("")  # 空行分隔
        
        # 3. 检索到的上下文
        prompt_parts.append("相关案例信息：")
        prompt_parts.append(context)
        prompt_parts.append("")  # 空行分隔
        
        # 4. 当前查询
        prompt_parts.append("用户问题：")
        prompt_parts.append(query)
        prompt_parts.append("")  # 空行分隔
        
        # 5. 回答要求（再次强调）
        prompt_parts.append("请根据以上案例信息回答问题，要求：")
        prompt_parts.append("1. 准确引用案例中的具体信息")
        prompt_parts.append("2. 注明信息来源（部门、时间）")
        prompt_parts.append("3. 如果信息不足或不确定，如实说明")
        prompt_parts.append("4. 使用正式、专业的政务语言")
        
        return "\n".join(prompt_parts)
    
    async def generate_response(
        self, 
        query: str, 
        context: str, 
        history: List[Dict] = None,
        stream: bool = False
    ) -> Dict[str, any]:
        """
        生成回答（核心方法）
        
        Args:
            query: 用户查询
            context: 检索到的上下文
            history: 对话历史
            stream: 是否流式输出
        
        Returns:
            {
                "answer": str,           # 生成的回答
                "usage": Dict,           # token使用情况
                "model": str,            # 使用的模型
                "finish_reason": str,    # 结束原因
                "timestamp": str         # 生成时间
            }
        """
        start_time = datetime.now()
        
        try:
            # 构建Prompt
            prompt = self.build_rag_prompt(query, context, history)
            
            # 记录日志（生产环境可控制长度）
            logger.debug(f"生成Prompt长度: {len(prompt)}字符")
            
            # 调用Qwen API
            if stream:
                return await self._generate_stream(prompt)
            else:
                return await self._generate_once(prompt, start_time)
                
        except Exception as e:
            logger.error(f"❌ LLM生成失败: {e}")
            return {
                "answer": "抱歉，生成回答时出现错误，请稍后重试。",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _generate_once(self, prompt: str, start_time: datetime) -> Dict[str, any]:
        """一次性生成完整回答"""
        response = Generation.call(
            model=self.model_name,
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            result_format='message'  # 返回结构化消息
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
            stream=True,  # 启用流式
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
    
    def evaluate_response_quality(self, answer: str, context: str) -> Dict[str, float]:
        """
        简单评估回答质量（可扩展）
        
        Returns:
            质量评分字典
        """
        scores = {
            "relevance": 0.8,  # 相关性（可根据内容计算）
            "completeness": 0.7,  # 完整性
            "accuracy": 0.9,  # 准确性
            "formality": 0.8,  # 正式程度
        }
        
        # 简单启发式评分（后续可扩展为模型评估）
        if "根据以上案例" in answer or "参考" in answer:
            scores["groundedness"] = 0.8  # 基于上下文的程度
        
        if "部门" in answer and "时间" in answer:
            scores["attribution"] = 0.9  # 来源标注
        
        return scores


# 工具函数：获取服务实例
def get_llm_service() -> LLMService:
    """获取LLM服务单例实例"""
    return LLMService()


# 工具函数：生成完整RAG回答
async def generate_rag_response(
    query: str, 
    context: str, 
    history: List[Dict] = None
) -> Dict[str, any]:
    """
    快速生成RAG回答（便捷函数）
    
    Example:
        result = await generate_rag_response("雨露计划", context_text)
        print(result["answer"])
    """
    service = get_llm_service()
    return await service.generate_response(query, context, history)