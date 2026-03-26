"""
政务问政智能助手 - Streamlit 前端界面
"""

import streamlit as st
import requests
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
import uuid

# 页面配置
st.set_page_config(
    page_title="政务问政智能助手",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 尝试导入 session_manager，失败则使用内存模式
try:
    import sys
    sys.path.insert(0, str(__file__).rsplit("/", 1)[0])
    from session_manager import session_manager
    USE_SESSION_MANAGER = True
except Exception:
    USE_SESSION_MANAGER = False
    session_manager = None


# ==================== 常量配置 ====================

API_BASE_URL = "http://localhost:8000/api"

# 问政类型映射
REQUEST_TYPE_MAP = {
    "咨询": {"icon": "💬", "color": "#3498db"},
    "投诉": {"icon": "🔴", "color": "#e74c3c"},
    "建议": {"icon": "💡", "color": "#f39c12"},
    "求助": {"icon": "🆘", "color": "#9b59b6"},
    "未知": {"icon": "❓", "color": "#95a5a6"},
}

# 样式配置
USER_MSG_STYLE = """
<div style='background-color: #3498db; color: white; padding: 12px 16px;
            border-radius: 12px; margin: 8px 0; max-width: 80%; margin-left: auto;'>
<b>您</b>
</div>
"""

ASSISTANT_MSG_STYLE = """
<div style='background-color: #f8f9fa; color: #2c3e50; padding: 12px 16px;
            border-radius: 12px; margin: 8px 0; max-width: 80%;'>
<b>智能助手</b>
</div>
"""


# ==================== 工具函数 ====================

def init_session_state():
    """初始化 session state"""
    if "current_session_id" not in st.session_state:
        if USE_SESSION_MANAGER:
            # 创建新会话
            session_id = session_manager.create_session()
            st.session_state.current_session_id = session_id
        else:
            st.session_state.current_session_id = datetime.now().strftime("%Y%m%d%H%M%S")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "api_base_url" not in st.session_state:
        st.session_state.api_base_url = API_BASE_URL


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

    # 如果删除的是当前会话，切换到另一个
    if st.session_state.current_session_id == session_id:
        sessions = get_sessions()
        if sessions:
            st.session_state.current_session_id = sessions[0]["id"]
        else:
            create_new_session()
    st.rerun()


def switch_session(session_id: str):
    """切换会话"""
    st.session_state.current_session_id = session_id
    if USE_SESSION_MANAGER:
        session = session_manager.get_session(session_id)
        if session:
            st.session_state.messages = [
                {
                    "role": m["role"],
                    "content": m["content"],
                    "metadata": m.get("metadata", {})
                }
                for m in session.get("messages", [])
            ]
    st.rerun()


def send_message(query: str) -> Dict[str, Any]:
    """发送消息到后端 API"""
    try:
        response = requests.post(
            f"{st.session_state.api_base_url}/chat",
            json={
                "query": query,
                "session_id": st.session_state.current_session_id,
                "top_k": 5,
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"请求失败: {str(e)}")
        return {
            "answer": "抱歉，服务暂时不可用，请稍后重试。",
            "classification": {"type": "未知", "request_department": ""},
            "sources": [],
            "quality_score": 0.0,
            "work_order_id": None,
        }


def save_message(role: str, content: str, metadata: Optional[Dict] = None):
    """保存消息"""
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "metadata": metadata or {}
    })

    if USE_SESSION_MANAGER:
        session_manager.add_message(
            st.session_state.current_session_id,
            role,
            content,
            metadata
        )


def get_stats() -> Dict[str, Any]:
    """获取统计信息"""
    if USE_SESSION_MANAGER:
        return session_manager.get_stats()

    # 内存模式统计
    return {
        "total_sessions": 1,
        "total_messages": len(st.session_state.messages),
        "user_messages": sum(1 for m in st.session_state.messages if m.get("role") == "user"),
        "assistant_messages": sum(1 for m in st.session_state.messages if m.get("role") == "assistant"),
    }


# ==================== UI 组件 ====================

def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.title("🏛️ 政务问政助手")

        # 新建会话按钮
        if st.button("➕ 新建会话", use_container_width=True):
            create_new_session()

        st.divider()

        # 会话历史列表
        st.subheader("会话历史")

        sessions = get_sessions()
        if not sessions:
            st.info("暂无会话记录")

        for session in sessions:
            is_active = session["id"] == st.session_state.current_session_id

            col1, col2 = st.columns([4, 1])
            with col1:
                btn_label = f"● {session['title'][:15]}" if is_active else f"○ {session['title'][:15]}"
                if st.button(btn_label, key=f"session_{session['id']}", use_container_width=True):
                    switch_session(session["id"])

            with col2:
                if st.button("🗑️", key=f"del_{session['id']}", use_container_width=True):
                    delete_session(session["id"])

        st.divider()

        # 统计信息
        st.subheader("📊 统计概览")
        stats = get_stats()
        col1, col2 = st.columns(2)
        col1.metric("会话数", stats.get("total_sessions", 0))
        col2.metric("消息数", stats.get("total_messages", 0))

        st.divider()

        # 设置入口
        if st.button("⚙️ 设置", use_container_width=True):
            st.session_state.show_settings = True
            st.rerender()


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
        # 导出对话
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
        <div style='display: flex; justify-content: flex-end; margin: 8px 0;'>
            <div style='background-color: #3498db; color: white; padding: 12px 16px;
                        border-radius: 12px; max-width: 80%;'>
                <b>您</b><br>{content}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # AI 回复
        classification = metadata.get("classification", {})
        request_type = classification.get("type", "未知")
        request_dept = classification.get("request_department", "")
        type_info = REQUEST_TYPE_MAP.get(request_type, REQUEST_TYPE_MAP["未知"])

        # 构建消息头
        header_html = f"""
        <div style='background-color: #f8f9fa; padding: 12px; border-radius: 12px 12px 0 0; margin-bottom: -8px;'>
            <span style='background-color: {type_info["color"]}; color: white; padding: 4px 12px;
                         border-radius: 12px; font-size: 12px;'>
                {type_info["icon"]} {request_type}
            </span>
            {f'<span style="margin-left: 10px; color: #666;">📍 {request_dept}</span>' if request_dept else ''}
        </div>
        """

        st.markdown(header_html, unsafe_allow_html=True)

        with st.container():
            st.markdown(f"""
            <div style='background-color: #f8f9fa; padding: 12px 16px; border-radius: 0 0 12px 12px;'>
                {content}
            </div>
            """, unsafe_allow_html=True)

            # 显示质量评分
            quality_score = metadata.get("quality_score", 0)
            if quality_score > 0:
                st.caption(f"📈 质量评分: {quality_score:.2f}")

            # 显示工单状态
            work_order_id = metadata.get("work_order_id")
            if work_order_id:
                st.success(f"📝 工单已创建: {work_order_id}")

            # 显示检索来源
            sources = metadata.get("sources", [])
            if sources:
                with st.expander("📚 检索来源"):
                    for i, source in enumerate(sources[:3], 1):
                        st.markdown(f"""
                        **{i}. {source.get('title', '无标题')}**
                        - 部门: {source.get('department', '未知')}
                        - 时间: {source.get('time', '未知')}
                        - 相似度: {source.get('similarity', 0):.2f}
                        """)


def render_chat_area():
    """渲染聊天区域"""
    # 显示历史消息
    for msg in st.session_state.messages:
        render_message(msg)


def render_input_area():
    """渲染输入区域"""
    col1, col2 = st.columns([6, 1])

    with col1:
        query = st.text_input(
            "请输入您的问题...",
            key="query_input",
            placeholder="例如：雨露计划什么时候发放？",
            label_visibility="collapsed",
            on_change=handle_submit,
            args=()
        )

    with col2:
        if st.button("发送", type="primary", use_container_width=True):
            handle_submit()


def handle_submit():
    """处理消息提交"""
    query = st.session_state.get("query_input", "").strip()
    if not query:
        return

    # 保存用户消息
    save_message("user", query)

    # 清空输入框
    st.session_state.query_input = ""

    # 调用 API
    with st.spinner("智能助手正在思考中..."):
        response = send_message(query)

    # 保存助手回复
    save_message(
        "assistant",
        response.get("answer", "抱歉，服务暂时不可用。"),
        metadata={
            "classification": response.get("classification", {}),
            "quality_score": response.get("quality_score", 0),
            "work_order_id": response.get("work_order_id"),
            "sources": response.get("sources", []),
            "timestamp": response.get("timestamp", ""),
        }
    )

    st.rerun()


def render_settings():
    """渲染设置页面"""
    st.title("⚙️ 设置")

    # API 地址配置
    st.subheader("API 配置")
    api_url = st.text_input(
        "API 地址",
        value=st.session_state.api_base_url,
        help="后端 API 的基础地址"
    )

    if api_url != st.session_state.api_base_url:
        st.session_state.api_base_url = api_url
        st.success("API 地址已更新")

    # 测试连接
    if st.button("测试连接"):
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            if response.status_code == 200:
                st.success("✅ 连接成功！")
            else:
                st.error(f"❌ 连接失败: {response.status_code}")
        except Exception as e:
            st.error(f"❌ 连接失败: {str(e)}")

    st.divider()

    # 数据管理
    st.subheader("数据管理")

    if st.button("🗑️ 清除所有会话数据"):
        if USE_SESSION_MANAGER:
            sessions = get_sessions()
            for s in sessions:
                session_manager.delete_session(s["id"])
        st.session_state.messages = []
        st.success("数据已清除")
        st.rerun()

    st.divider()

    # 返回主页
    if st.button("← 返回主页"):
        st.session_state.show_settings = False
        st.rerun()


def export_chat():
    """导出对话为 Markdown"""
    md_content = f"# 政务问政对话记录\n\n"
    md_content += f"导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    for msg in st.session_state.messages:
        role = "用户" if msg["role"] == "user" else "智能助手"
        content = msg["content"]
        md_content += f"## {role}\n\n{content}\n\n"

        metadata = msg.get("metadata", {})
        if metadata:
            classification = metadata.get("classification", {})
            if classification:
                md_content += f"- 类型: {classification.get('type', '未知')}\n"
                if classification.get("request_department"):
                    md_content += f"- 部门: {classification.get('request_department')}\n"

            if metadata.get("quality_score"):
                md_content += f"- 质量评分: {metadata.get('quality_score')}\n"

            if metadata.get("work_order_id"):
                md_content += f"- 工单ID: {metadata.get('work_order_id')}\n"

        md_content += "\n"

    st.download_button(
        label="📥 下载 Markdown",
        data=md_content,
        file_name=f"对话记录_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown"
    )


# ==================== 主程序 ====================

def main():
    """主程序入口"""
    # 初始化
    init_session_state()

    # 检查是否显示设置页面
    if st.session_state.get("show_settings", False):
        render_settings()
        return

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