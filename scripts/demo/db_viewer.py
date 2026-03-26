import streamlit as st
import pandas as pd
from pymilvus import MilvusClient
import os
import sys

sys.path.append(os.getcwd())

from src.config.setting import settings

# ================= 配置 =================
st.set_page_config(layout="wide", page_title="Milvus 数据库查看器")
DB_PATH = str(settings.vectordb.db_path)
COLLECTION_NAME = settings.vectordb.gov_cases_collection_name

# ================= 侧边栏：连接数据库 =================
st.sidebar.title("🗄️ 数据库连接")
if not os.path.exists(DB_PATH):
    st.error(f"❌ 找不到数据库文件: {DB_PATH}")
    st.stop()

try:
    # 初始化连接
    client = MilvusClient(DB_PATH)
    st.sidebar.success(f"✅ 已连接: {os.path.basename(DB_PATH)}")
except Exception as e:
    st.error(f"连接失败: {e}")
    st.stop()

# ================= 主界面 =================
st.title("🔎 GovPulse 数据库可视化")

# 1. 集合概览
st.subheader("1. 集合概览")
if client.has_collection(COLLECTION_NAME):
    res = client.query(collection_name=COLLECTION_NAME, filter="", output_fields=["count(*)"])
    total_count = res[0]["count(*)"]
    
    col1, col2 = st.columns(2)
    col1.metric("集合名称", COLLECTION_NAME)
    col2.metric("数据总条数", total_count)
else:
    st.warning(f"集合 {COLLECTION_NAME} 不存在！")
    st.stop()

# 2. 数据浏览
st.subheader("2. 数据浏览")

# 分页控制
limit = st.sidebar.slider("每页显示条数", 10, 100, 20)
offset = st.sidebar.number_input("偏移量 (Offset)", min_value=0, value=0, step=limit)

# 拉取数据
# filter="" 表示匹配所有
# 不拉取 vector 字段，因为它太长了，显示出来没意义且卡顿
data = client.query(
    collection_name=COLLECTION_NAME,
    filter="",
    limit=limit,
    offset=offset,
    output_fields=["text", "department", "metadata"]
)

if data:
    # 处理数据以便展示
    df_list = []
    for item in data:
        meta = item.get('metadata', {})
        row = {
            "ID (Hash)": item.get('id'), # 自动生成的 ID
            "Doc ID (MD5)": meta.get('doc_id', 'N/A'),
            "部门": item.get('department'),
            "RAG 上下文 (Text)": item.get('text'),
            "原始问题": meta.get('question', ''),
            "原始回答": meta.get('answer', '')
        }
        df_list.append(row)
    
    df = pd.DataFrame(df_list)
    st.dataframe(df, use_container_width=True)
else:
    st.info("当前页没有数据。")

# 3. 简单的 ID 查询
st.subheader("3. 调试：ID 查询")
search_id = st.text_input("输入 doc_id (MD5) 进行查询")
if search_id:
    # JSON 里的字段需要用 json_contains 或者特定语法，Milvus Lite 支持基础 filter
    # 注意：metadata["doc_id"] 这种写法取决于 Milvus 版本，
    # 简单的做法是遍历或者依赖之前的 embedding 检索。
    # 这里演示 Metadata 过滤 (Milvus Lite 对 JSON 过滤支持有限，可能需要特定语法)
    
    st.caption("注：Milvus Lite 对 JSON 字段的直接 SQL 过滤支持可能不完善，建议使用代码脚本进行精确查找。")
    
    # 尝试过滤 (针对动态字段或特定Schema)
    try:
        res = client.query(
            collection_name=COLLECTION_NAME, 
            filter=f'metadata["doc_id"] == "{search_id}"',
            output_fields=["text"]
        )
        if res:
            st.success("找到数据！")
            st.json(res[0])
        else:
            st.warning("未找到该 ID")
    except Exception as e:
        st.error(f"查询语法错误或不支持: {e}")