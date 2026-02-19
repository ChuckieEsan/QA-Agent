"""
LLM 服务抽象基类
定义统一的 LLM 服务接口
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, AsyncGenerator, Any


class BaseLLMService(ABC):
    """
    LLM 服务抽象基类

    所有 LLM 服务实现都应该继承此类
    """

    @abstractmethod
    async def analyze_query_intent(
        self,
        query: str,
        history: Optional[List[Dict]] = None
    ) -> Any:  # AgentDecision
        """
        分析查询意图

        Args:
            query: 用户查询
            history: 对话历史（可选）

        Returns:
            AgentDecision 决策结果
        """
        pass

    @abstractmethod
    async def generate_response(
        self,
        query: str,
        context: str,
        history: Optional[List[Dict]] = None,
        decision: Optional[Any] = None,  # AgentDecision
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        生成回答

        Args:
            query: 用户查询
            context: 检索上下文
            history: 对话历史（可选）
            decision: Agent 决策（可选）
            stream: 是否流式生成

        Returns:
            生成结果字典
        """
        pass

    @abstractmethod
    async def validate_answer_quality(
        self,
        answer: str,
        query: str,
        context: str
    ) -> Dict[str, Any]:
        """
        验证回答质量

        Args:
            answer: 生成的回答
            query: 用户查询
            context: 检索上下文

        Returns:
            质量校验结果
        """
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """
        初始化 LLM 服务资源
        """
        pass
