"""
GovPulse 前端入口文件
提供多页面导航功能
"""

import streamlit as st

# 页面配置
st.set_page_config(
    page_title="政务问政智能助手",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 页面导航
st.navigation([
    st.Page("pages/01_chat.py", title="智能问答", icon="💬", default=True),
    st.Page("pages/02_knowledge.py", title="知识库管理", icon="📚"),
    st.Page("pages/03_analytics.py", title="数据分析", icon="📊"),
    st.Page("pages/04_settings.py", title="系统设置", icon="⚙️"),
]).run()