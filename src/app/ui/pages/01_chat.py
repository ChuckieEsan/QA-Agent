"""
智能问答页面
"""

import json
import asyncio
import streamlit as st
from datetime import datetime
from typing import Dict, Any, List

from app.agents import ainvoke
from app.ui.session_manager import session_manager

USE_SESSION_MANAGER = True


# ==================== 常量配置 ====================

# 问政类型映射
REQUEST_TYPE_MAP = {
    "咨询": {"icon": "💬", "color": "#3498db"},
    "投诉": {"icon": "🔴", "color": "#e74c3c"},
    "建议": {"icon": "💡", "color": "#f39c12"},
    "求助": {"icon": "🆘", "color": "#9b59b6"},
    "未知": {"icon": "❓", "color": "#95a5a6"},
}


# ==================== 工具函数 ====================

def init_session_state():
    """初始化 session state"""
    if "current_session_id" not in st.session_state:
        # 检查是否有现成的空会话，避免重复创建
        if USE_SESSION_MANAGER:
            sessions = session_manager.get_all_sessions()
            # 找到最新的空会话
            empty_session = None
            for s in sessions:
                if not s.get("messages"):
                    empty_session = s
                    break
            if empty_session:
                st.session_state.current_session_id = empty_session.get("session_id") or empty_session.get("id")
            else:
                # 没有空会话则创建新的
                session_id = session_manager.create_session()
                st.session_state.current_session_id = session_id
        else:
            st.session_state.current_session_id = datetime.now().strftime("%Y%m%d%H%M%S")

    if "messages" not in st.session_state:
        # 从会话管理器加载消息
        if USE_SESSION_MANAGER:
            session = session_manager.get_session(st.session_state.current_session_id)
            if session:
                st.session_state.messages = session.get("messages", [])
            else:
                st.session_state.messages = []
        else:
            st.session_state.messages = []


def get_sessions() -> List[Dict[str, Any]]:
    """获取所有会话"""
    if USE_SESSION_MANAGER:
        return session_manager.get_all_sessions()
    return []


def create_new_session() -> str:
    """创建新会话"""
    if USE_SESSION_MANAGER:
        session_id = session_manager.create_session()
    else:
        session_id = datetime.now().strftime("%Y%m%d%H%M%S")
    st.session_state.current_session_id = session_id
    st.session_state.messages = []
    st.rerun()


def delete_session(session_id: str):
    """删除会话"""
    if USE_SESSION_MANAGER:
        session_manager.delete_session(session_id)
    st.rerun()


def send_message(query: str) -> Dict[str, Any]:
    """发送消息到后端"""
    try:
        # 使用 ainvoke 直接调用
        response = asyncio.run(
            ainvoke(query, session_id=st.session_state.current_session_id)
        )

        # 转换 AgentResponse 为字典
        return {
            "answer": response.final_reply,
            "classification": response.classification,
            "confidence_score": response.confidence_score,
            "work_order_id": response.work_order_id,
        }
    except Exception as e:
        st.error(f"调用失败: {str(e)}")
        return {
            "answer": f"抱歉，服务暂时不可用: {str(e)}",
            "classification": {},
            "confidence_score": 0.0,
            "work_order_id": None,
        }


def export_chat():
    """导出对话"""
    messages = st.session_state.messages
    if not messages:
        st.warning("没有可导出的对话")
        return

    chat_data = {
        "session_id": st.session_state.current_session_id,
        "export_time": datetime.now().isoformat(),
        "messages": messages,
    }

    st.download_button(
        label="📥 下载对话",
        data=json.dumps(chat_data, ensure_ascii=False, indent=2),
        file_name=f"chat_{st.session_state.current_session_id}.json",
        mime="text/json",
    )


# ==================== 页面渲染 ====================

def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.title("🏛️ 政务问政")

        # 新建会话
        if st.button("➕ 新建会话", use_container_width=True):
            create_new_session()

        st.divider()

        # 会话列表
        st.subheader("历史会话")

        sessions = get_sessions()
        if sessions:
            for i, s in enumerate(sessions):
                session_id = s.get("session_id") or s.get("id", "")
                # 跳过无效会话
                if not session_id:
                    continue
                title = s.get("title", "未命名会话")[:20]
                is_active = session_id == st.session_state.current_session_id

                # 选中状态用不同样式
                button_type = "primary" if is_active else "secondary"
                col1, col2 = st.columns([4, 1])
                with col1:
                    if st.button(
                        f"📝 {title}",
                        key=f"session_{session_id}",
                        type=button_type,
                        use_container_width=True,
                    ):
                        st.session_state.current_session_id = session_id
                        st.session_state.messages = s.get("messages", [])
                        st.rerun()
                with col2:
                    if st.button("🗑️", key=f"del_{session_id}"):
                        delete_session(session_id)
        else:
            st.info("暂无历史会话")

        st.divider()

        # 统计信息
        st.subheader("📊 统计")
        stats = get_stats() if USE_SESSION_MANAGER else {"total_sessions": 0, "total_messages": 0}
        col1, col2 = st.columns(2)
        col1.metric("会话数", stats.get("total_sessions", 0))
        col2.metric("消息数", stats.get("total_messages", 0))


def render_chat_header():
    """渲染聊天头部"""
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

    with col2:
        if st.button("🔄 转人工", use_container_width=True):
            st.toast("正在转接人工服务...")

    with col3:
        if st.button("🗑️ 清除历史", use_container_width=True):
            if USE_SESSION_MANAGER:
                session_manager.clear_session(st.session_state.current_session_id)
            st.session_state.messages = []
            st.rerun()

    with col4:
        if st.button("📥 导出", use_container_width=True):
            export_chat()

    st.divider()


def render_message(msg: Dict[str, Any]):
    """渲染单条消息"""
    role = msg.get("role", "assistant")
    content = msg.get("content", "")
    metadata = msg.get("metadata", {})

    if role == "user":
        st.markdown(f"""
        <div style='display: flex; justify-content: flex-end; margin: 4px 0;'>
            <div style='background-color: #3498db; color: white; padding: 8px 12px;
                        border-radius: 10px; max-width: 80%; font-size: 14px;'>
                <b>您</b><br>{content}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # AI 回复
        with st.container():
            col1, col2 = st.columns([6, 1])
            with col1:
                st.markdown(f"""
                <div style='background-color: #f8f9fa; color: #2c3e50; padding: 8px 12px;
                            border-radius: 10px; margin: 4px 0; max-width: 80%; font-size: 14px;'>
                    <b>🏛️ 智能助手</b><br>
                    {content}
                </div>
                """, unsafe_allow_html=True)
            with col2:
                # 显示分类信息
                if metadata.get("classification"):
                    req_type = metadata["classification"].get("request_type", "未知")
                    req_type_info = REQUEST_TYPE_MAP.get(req_type, REQUEST_TYPE_MAP["未知"])
                    st.caption(f"{req_type_info['icon']} {req_type}")


def render_chat_area():
    """渲染聊天区域"""
    for msg in st.session_state.messages:
        render_message(msg)


def render_input_area():
    """渲染输入区域"""
    # 预置问题
    preset_questions = [
        "灵活就业人员如何缴纳社保？",
        "如何办理居住证？",
        "社保转移怎么办理？",
    ]

    # 处理预置问题按钮点击 - 直接发送消息
    for i, q in enumerate(preset_questions):
        if st.button(q, key=f"preset_{i}", use_container_width=True):
            # 添加用户消息
            st.session_state.messages.append({
                "role": "user",
                "content": q,
                "timestamp": datetime.now().isoformat(),
            })

            # 调用后端
            with st.spinner("🤔 智能助手思考中..."):
                response = send_message(q)

            # 添加 AI 消息
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["answer"],
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "classification": response.get("classification", {}),
                    "confidence_score": response.get("confidence_score", 0.0),
                    "work_order_id": response.get("work_order_id"),
                },
            })

            # 保存到会话管理器
            if USE_SESSION_MANAGER:
                session_manager.add_message(
                    st.session_state.current_session_id,
                    "user", q,
                )
                session_manager.add_message(
                    st.session_state.current_session_id,
                    "assistant", response["answer"],
                )

            st.rerun()

    st.divider()

    # 输入框 - 更紧凑
    if "input_text" not in st.session_state:
        st.session_state.input_text = ""

    query = st.text_area(
        "请输入您的问题：",
        value=st.session_state.input_text,
        height=60,
        key="query_input",
        label_visibility="collapsed",
    )

    col1, col2 = st.columns([6, 1])
    with col1:
        submit = st.button("🚀 发送", type="primary", use_container_width=True)
    with col2:
        clear = st.button("🧹 清除", use_container_width=True)

    if clear:
        st.session_state.input_text = ""
        st.rerun()

    if submit and query.strip():
        # 添加用户消息
        st.session_state.messages.append({
            "role": "user",
            "content": query.strip(),
            "timestamp": datetime.now().isoformat(),
        })

        # 清空输入
        st.session_state.input_text = ""

        # 调用后端
        with st.spinner("🤔 智能助手思考中..."):
            response = send_message(query.strip())

        # 添加 AI 消息
        st.session_state.messages.append({
            "role": "assistant",
            "content": response["answer"],
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "classification": response.get("classification", {}),
                "confidence_score": response.get("confidence_score", 0.0),
                "work_order_id": response.get("work_order_id"),
            },
        })

        # 保存到会话管理器
        if USE_SESSION_MANAGER:
            session_manager.add_message(
                st.session_state.current_session_id,
                "user", query.strip(),
            )
            session_manager.add_message(
                st.session_state.current_session_id,
                "assistant", response["answer"],
            )

        st.rerun()


def get_stats() -> Dict[str, Any]:
    """获取统计信息"""
    if USE_SESSION_MANAGER:
        return session_manager.get_stats()
    return {"total_sessions": 0, "total_messages": 0}


# ==================== 主程序 ====================

def main():
    """主程序入口"""
    # 初始化
    init_session_state()

    # 渲染侧边栏
    render_sidebar()

    # 渲染主区域
    st.title("🏛️ 政务问政智能助手")

    # 渲染聊天头部
    render_chat_header()

    # 渲染聊天区域
    render_chat_area()

    # 渲染输入区域
    render_input_area()


if __name__ == "__main__":
    main()