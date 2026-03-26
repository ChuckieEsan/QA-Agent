"""
会话管理器 - 负责会话历史的本地 JSON 持久化
"""

import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

# 会话文件存储路径
SESSION_FILE = Path(__file__).parent / "sessions.json"


class SessionManager:
    """会话管理器"""

    def __init__(self, session_file: Path = SESSION_FILE):
        self.session_file = session_file
        self._ensure_file()

    def _ensure_file(self):
        """确保会话文件存在"""
        if not self.session_file.exists():
            self._save_sessions({})

    def _load_sessions(self) -> Dict[str, Any]:
        """加载所有会话"""
        try:
            with open(self.session_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def _save_sessions(self, sessions: Dict[str, Any]):
        """保存所有会话"""
        with open(self.session_file, "w", encoding="utf-8") as f:
            json.dump(sessions, f, ensure_ascii=False, indent=2)

    def create_session(self, title: Optional[str] = None) -> str:
        """创建新会话"""
        sessions = self._load_sessions()
        session_id = datetime.now().strftime("%Y%m%d%H%M%S")

        sessions[session_id] = {
            "id": session_id,
            "title": title or f"新会话 {len(sessions) + 1}",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "messages": [],
        }
        self._save_sessions(sessions)
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取指定会话"""
        sessions = self._load_sessions()
        return sessions.get(session_id)

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """获取所有会话（按更新时间倒序）"""
        sessions = self._load_sessions()
        session_list = list(sessions.values())
        return sorted(session_list, key=lambda x: x.get("updated_at", ""), reverse=True)

    def add_message(self, session_id: str, role: str, content: str, metadata: Optional[Dict] = None):
        """添加消息到会话"""
        sessions = self._load_sessions()
        if session_id not in sessions:
            return

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        if metadata:
            message["metadata"] = metadata

        sessions[session_id]["messages"].append(message)
        sessions[session_id]["updated_at"] = datetime.now().isoformat()

        # 如果是第一条用户消息，更新会话标题
        if len(sessions[session_id]["messages"]) == 1 and role == "user":
            # 使用消息前20个字符作为标题
            sessions[session_id]["title"] = content[:20] + "..." if len(content) > 20 else content

        self._save_sessions(sessions)

    def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        sessions = self._load_sessions()
        if session_id in sessions:
            del sessions[session_id]
            self._save_sessions(sessions)
            return True
        return False

    def clear_session(self, session_id: str) -> bool:
        """清除会话消息"""
        sessions = self._load_sessions()
        if session_id in sessions:
            sessions[session_id]["messages"] = []
            sessions[session_id]["updated_at"] = datetime.now().isoformat()
            self._save_sessions(sessions)
            return True
        return False

    def update_session_title(self, session_id: str, title: str) -> bool:
        """更新会话标题"""
        sessions = self._load_sessions()
        if session_id in sessions:
            sessions[session_id]["title"] = title
            sessions[session_id]["updated_at"] = datetime.now().isoformat()
            self._save_sessions(sessions)
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        sessions = self._load_sessions()
        total_messages = sum(len(s["messages"]) for s in sessions.values())

        # 统计消息类型
        user_messages = sum(1 for s in sessions.values() for m in s["messages"] if m.get("role") == "user")
        assistant_messages = sum(1 for s in sessions.values() for m in s["messages"] if m.get("role") == "assistant")

        return {
            "total_sessions": len(sessions),
            "total_messages": total_messages,
            "user_messages": user_messages,
            "assistant_messages": assistant_messages,
        }


# 全局单例
session_manager = SessionManager()