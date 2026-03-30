"""
数据分析页面
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="数据分析",
    page_icon="📊",
)

st.title("📊 数据分析")

# 时间范围选择
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("开始日期")
with col2:
    end_date = st.date_input("结束日期")

st.divider()

# 核心指标
st.subheader("📈 核心指标")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("总咨询量", "12,345", "+15%")
with col2:
    st.metric("智能解答率", "87.5%", "+2.3%")
with col3:
    st.metric("平均响应时间", "2.3s", "-0.5s")
with col4:
    st.metric("用户满意度", "4.6/5", "+0.2")

st.divider()

# 图表区域
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 诉求类型分布")
    # 模拟数据
    data = {
        "类型": ["咨询", "投诉", "建议", "求助"],
        "数量": [4500, 2300, 1800, 1200],
    }
    df = pd.DataFrame(data)
    st.bar_chart(df.set_index("类型"))

with col2:
    st.subheader("📈 趋势变化")
    # 模拟数据
    dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
    values = np.random.randint(100, 500, size=30)
    chart_data = pd.DataFrame({"日期": dates, "咨询量": values})
    st.line_chart(chart_data.set_index("日期"))

st.divider()

# 详细数据表格
st.subheader("📋 详细数据")

# 模拟详细数据
detail_data = {
    "日期": ["2024-03-01", "2024-03-02", "2024-03-03", "2024-03-04", "2024-03-05"],
    "咨询量": [156, 189, 201, 178, 165],
    "解答率": ["88%", "85%", "90%", "87%", "89%"],
    "满意度": ["4.5", "4.6", "4.7", "4.5", "4.6"],
}

df_detail = pd.DataFrame(detail_data)
st.dataframe(df_detail, use_container_width=True)