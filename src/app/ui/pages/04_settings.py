"""
系统设置页面
"""

import streamlit as st

st.set_page_config(
    page_title="系统设置",
    page_icon="⚙️",
)

st.title("⚙️ 系统设置")

# LLM 配置
st.header("🤖 LLM 配置")

col1, col2 = st.columns(2)

with col1:
    llm_provider = st.selectbox(
        "模型提供商",
        ["DeepSeek", "Qwen", "Ollama"],
        index=0,
    )

with col2:
    model_name = st.selectbox(
        "模型名称",
        ["deepseek-chat", "qwen-turbo", "llama2"],
        index=0,
    )

st.divider()

# RAG 配置
st.header("📚 RAG 配置")

col1, col2 = st.columns(2)

with col1:
    top_k = st.slider(
        "检索结果数量",
        min_value=1,
        max_value=20,
        value=5,
        help="每次检索返回的结果数量",
    )

with col2:
    similarity_threshold = st.slider(
        "相似度阈值",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="低于此阈值的结果将被过滤",
    )

st.divider()

# 验证器配置
st.header("✅ 验证器配置")

col1, col2 = st.columns(2)

with col1:
    accuracy_threshold = st.slider(
        "准确性阈值",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.1,
        help="答案校验的准确性阈值",
    )

with col2:
    enable_validator = st.toggle(
        "启用答案校验",
        value=True,
        help="是否启用 RAG Triad 校验",
    )

st.divider()

# MCP 配置
st.header("🔗 MCP 配置")

mcp_enabled = st.toggle(
    "启用 MCP 协议",
    value=True,
    help="是否启用 MCP 协议对接下游服务",
)

if mcp_enabled:
    mcp_server_url = st.text_input(
        "MCP 服务器地址",
        value="http://localhost:8080",
    )

st.divider()

# 保存按钮
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    if st.button("💾 保存设置", type="primary", use_container_width=True):
        st.success("设置已保存")

with col2:
    if st.button("🔄 恢复默认", use_container_width=True):
        st.info("已恢复默认设置")

with col3:
    if st.button("🔧 高级选项", use_container_width=True):
        st.session_state.show_advanced = True
        st.rerun()

# 高级选项
if st.session_state.get("show_advanced", False):
    with st.expander("🔧 高级选项", expanded=True):
        st.write("高级配置选项...")

        col1, col2 = st.columns(2)
        with col1:
            log_level = st.selectbox(
                "日志级别",
                ["DEBUG", "INFO", "WARNING", "ERROR"],
                index=1,
            )
        with col2:
            max_tokens = st.number_input(
                "最大 Token 数",
                min_value=100,
                max_value=10000,
                value=2000,
            )

        enable_cache = st.toggle(
            "启用响应缓存",
            value=True,
        )

        st.info("高级选项更改后需要重启服务生效")