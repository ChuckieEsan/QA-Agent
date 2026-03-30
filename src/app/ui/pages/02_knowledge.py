"""
知识库管理页面
"""

import streamlit as st

st.set_page_config(
    page_title="知识库管理",
    page_icon="📚",
)

st.title("📚 知识库管理")

st.info("该功能正在开发中...")

# 示例：显示当前知识库状态
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("政策文档", "1,234")

with col2:
    st.metric("历史案例", "5,678")

with col3:
    st.metric("权责清单", "890")

st.divider()

st.subheader("📤 数据导入")

uploaded_file = st.file_uploader(
    "选择要上传的文件",
    type=["xlsx", "csv", "json"],
    help="支持 Excel、CSV、JSON 格式"
)

if uploaded_file:
    st.success(f"已选择文件: {uploaded_file.name}")
    if st.button("开始导入"):
        with st.spinner("导入中..."):
            st.info("导入功能正在开发中")

st.divider()

st.subheader("🔍 数据预览")

# 示例表格
import pandas as pd

data = {
    "文档ID": ["DOC001", "DOC002", "DOC003"],
    "标题": ["社保政策解读", "居住证办理指南", "公积金提取条件"],
    "类型": ["政策", "指南", "政策"],
    "更新时间": ["2024-01-15", "2024-02-20", "2024-03-01"],
}

df = pd.DataFrame(data)
st.dataframe(df, use_container_width=True)