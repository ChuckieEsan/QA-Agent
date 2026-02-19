"""
对话记忆组件
简单的内存对话记忆实现
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from src.app.components.memory.base_memory import BaseMemory
from src.app.infra.utils.logger import get_logger

logger = get_logger(__name__)


class ConversationMemory(BaseMemory):
    """
    对话记忆组件

    使用内存存储对话历史
    """

    def __init__(self):
        self.messages: List[Dict[str, any]] = []
        self.created_at: datetime = datetime.now()
        self.last_updated: datetime = datetime.now()
        logger.info("✅ Conversation Memory 初始化完成")

    def add_message(self, message: Dict[str, any]) -> None:
        """
        添加消息到记忆

        Args:
            message: 消息字典，格式：{"role": "user|assistant", "content": str}
        """
        if "role" not in message or "content" not in message:
            raise ValueError("消息必须包含 'role' 和 'content' 字段")

        self.messages.append(message)
        self.last_updated = datetime.now()
        logger.debug(f"💾 添加消息: {message['role']}")

    def get_messages(self, limit: Optional[int] = None) -> List[Dict[str, any]]:
        """
        获取记忆中的消息

        Args:
            limit: 返回的消息数量限制（None 表示全部）

        Returns:
            消息列表
        """
        if limit is None:
            return self.messages.copy()
        return self.messages[-limit:].copy()

    def clear(self) -> None:
        """清空记忆"""
        self.messages = []
        self.last_updated = datetime.now()
        logger.info("🧹 记忆已清空")

    def get_context(self, max_tokens: Optional[int] = None) -> str:
        """
        获取对话上下文（用于 Prompt 构建）

        Args:
            max_tokens: 最大 token 数限制

        Returns:
            格式化的对话上下文字符串
        """
        if not self.messages:
            return ""

        # 构建上下文
        context_parts = ["## 对话历史"]
        for msg in self.messages[-5:]:  # 最多显示最近 5 条
            role = "用户" if msg["role"] == "user" else "助手"
            context_parts.append(f"{role}: {msg['content']}")

        context_parts.append("")
        return "\n".join(context_parts)

    def save(self, path: str) -> None:
        """
        保存记忆到文件

        Args:
            path: 保存路径
        """
        import json
        import os

        os.makedirs(os.path.dirname(path), exist_ok=True)

        data = {
            "messages": self.messages,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat()
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"💾 记忆已保存到: {path}")

    def load(self, path: str) -> None:
        """
        从文件加载记忆

        Args:
            path: 加载路径
        """
        import json
        import os

        if not os.path.exists(path):
            logger.warning(f"⚠️  文件不存在: {path}")
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.messages = data["messages"]
        self.created_at = datetime.fromisoformat(data["created_at"])
        self.last_updated = datetime.fromisoformat(data.get("last_updated", data["created_at"]))

        logger.info(f"📂 记忆已从 {path} 加载，共 {len(self.messages)} 条消息")

    def get_stats(self) -> Dict[str, any]:
        """
        获取记忆统计信息

        Returns:
            统计信息字典
        """
        user_count = sum(1 for msg in self.messages if msg["role"] == "user")
        assistant_count = sum(1 for msg in self.messages if msg["role"] == "assistant")

        # 粗略估算 token 数（按字符数 / 4）
        total_tokens = sum(len(msg["content"]) for msg in self.messages) // 4

        return {
            "total_messages": len(self.messages),
            "user_messages": user_count,
            "assistant_messages": assistant_count,
            "total_tokens": total_tokens,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat()
        }
