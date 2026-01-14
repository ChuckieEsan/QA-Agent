# 面向电子政务的智能 RAG 问答系统深度实施方案
**——基于 Qwen API 与端侧混合检索架构**

## 1. 系统架构设计 (System Architecture)

### 1.1 核心设计理念
*   **端云协同 (Edge-Cloud Synergy)**：
    *   **本地 (RTX 4050)**：负责高频、低延时的特征提取（Embedding）和 精准排序（Rerank）。利用 GPU 加速向量运算。
    *   **云端 (Aliyun)**：负责重逻辑、高显存消耗的意图理解（Routing）和 答案生成（Generation）。利用 Qwen 模型强大的中文推理能力。
*   **流水线化 (Pipelined)**：将检索过程拆解为 `Query 理解 -> 粗排召回 -> 精排过滤 -> 生成` 四个解耦环节。

### 1.2 技术栈选型表

| 模块          | 组件/算法  | 具体选型                           | 部署位置 | 显存预估 |
| :------------ | :--------- | :--------------------------------- | :------- | :------- |
| **LLM**       | 大语言模型 | **Qwen-Turbo / Qwen-Plus** (API)   | 阿里云   | -        |
| **Embedding** | 向量模型   | **BAAI/bge-m3** (FP16)             | 本地 GPU | ~1.5 GB  |
| **Indexing**  | 向量数据库 | **Milvus Lite** (Python库)         | 本地 RAM | -        |
| **Retrieval** | 稀疏检索   | **BM25** (`rank_bm25` 库)          | 本地 CPU | -        |
| **Reranking** | 交叉编码器 | **BAAI/bge-reranker-v2-m3**        | 本地 GPU | ~1.2 GB  |
| **Framework** | 编排框架   | **Python (原生开发)** 或 LangChain | 本地     | -        |

---

## 2. 详细技术实现方案

### 第一阶段：数据工程 (离线处理)

这是提升系统效果的基石。解决“用户口语”与“政务公文”不匹配的问题。

#### 2.1 数据清洗与结构化
*   **原始数据**：`{问题, 官方回复, 部门}`
*   **目标结构**：我们需要构建一个更丰富的 `Document` 对象。

#### 2.2 数据增强 (Data Augmentation) —— 关键创新点
利用 Qwen API，为每条政务案例生成“假设性用户提问”。
*   **逻辑**：官方回复通常很正式，直接用来做向量匹配效果不好。我们需要把官方回复“翻译”成用户可能问的问题。
*   **Qwen Prompt**:
    ```text
    你是一名政务数据分析师。请阅读以下“群众诉求”和“官方回复”：
    【诉求】：{content}
    【回复】：{reply}
    
    请生成 3 个不同风格的“模拟用户提问”，要求：
    1. 一个口语化，情绪激动（如“为什么没人管...”）。
    2. 一个关键词化，简短（如“XX办理流程”）。
    3. 一个标准化，语义完整。
    
    输出格式：["提问1", "提问2", "提问3"]
    ```
*   **处理结果**：将生成的 3 个问题 + 原始问题，拼接成一段文本，作为该案例的 `indexing_text`（索引文本）。

---

### 第二阶段：索引构建 (离线入库)

放弃 K-Means，使用 HNSW 索引。

#### 2.3 向量数据库 Schema 设计 (Milvus Lite)
Milvus Lite 允许你在 Python 脚本中直接运行向量库，生成一个 `milvus_demo.db` 文件，非常适合毕设。

**Collection Schema 定义**：
```python
from pymilvus import FieldSchema, CollectionSchema, DataType

fields = [
    # 主键 ID
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    # 原始案例 ID (关联回你的原始数据)
    FieldSchema(name="case_id", dtype=DataType.VARCHAR, max_length=64),
    # 部门 (用于元数据过滤)
    FieldSchema(name="department", dtype=DataType.VARCHAR, max_length=64),
    # 向量 (BGE-M3 维度为 1024)
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
    # 原始内容 (用于生成时提取，建议存部分或存外部引用)
    FieldSchema(name="content_summary", dtype=DataType.VARCHAR, max_length=2048)
]
schema = CollectionSchema(fields, "政务案例索引")
```

#### 2.4 索引参数 (HNSW)
HNSW 是目前最快的向量索引算法。
*   **Index Type**: `HNSW`
*   **Metric Type**: `COSINE` (余弦相似度)
*   **Params**: `{'M': 16, 'efConstruction': 200}`
    *   *解释*：`M` 是图连接数，`efConstruction` 决定建索引精度。这两个参数在 4050 上运行毫无压力。

---

### 第三阶段：在线检索流水线 (核心逻辑)

这是用户提问后的实时处理流程。

#### 3.1 模块一：查询理解 (Query Understanding)
*   **意图路由 (Routing)**：调用 Qwen-Turbo。
    *   *Prompt*: `用户问题：“{query}”。请判断该问题最可能涉及的职能部门（如：市医保局, 市教育局, 市住建局...）。如果涉及多个，请列出。返回 Python List 格式。`
    *   *输出*: `['市医保局']` -> 转化为 Milvus 的过滤条件 `expr = "department in ['市医保局']"`。
*   **查询重写 (Query Expansion)**：
    *   如果用户问题太短（如“社保”），调用 Qwen 改写为“社保缴纳与报销相关政策查询”，丰富语义。

#### 3.2 模块二：混合检索 (Hybrid Search)
为了解决“语义相似但实体不同”的问题（如“红星路”vs“红旗路”）。

1.  **向量检索 (Dense Retrieval)**:
    *   模型：本地加载 `BGE-M3`。
    *   操作：`query_vector = model.encode(user_query)`
    *   Milvus 查询：`search(data=[query_vector], limit=50, expr=dept_filter)`
    *   结果：获得 Top-50 候选集 A。
2.  **关键词检索 (Sparse Retrieval)**:
    *   工具：`rank_bm25` (内存中运行，速度极快)。
    *   操作：对 Top-50 候选集 A 再进行一次 BM25 打分？(或者全库 BM25，取决于数据量。推荐：**对 Milvus 召回的 Top-100 做 BM25 加权**，这样最省内存)。
3.  **加权融合**:
    *   `Final_Score = 0.7 * Vector_Score + 0.3 * BM25_Score` (权重可调)。

#### 3.3 模块三：重排序 (Reranking) —— **精度提升的关键**
向量检索只看“大概像不像”，重排序看“逻辑对不对”。

*   **模型**：本地加载 `BGE-Reranker-v2-m3`。
*   **输入**：Pairs `[(用户问题, 候选案例1), (用户问题, 候选案例2), ...]` (Top 50)。
*   **输出**：每个 Pair 的相关性得分 (0~1)。
*   **操作**：按得分降序排列，截取 **Top-5**。
*   **解决痛点**：此步骤能有效识别出虽然涉及“社保”关键词，但具体政策不符的错误案例。

---

### 第四阶段：回答生成 (Generation)

#### 3.4 提示词工程 (Prompt Engineering)
利用 Qwen-Plus 强大的上下文理解能力，处理多部门职责交叉。

*   **Prompt 模板**:
```text
你是一名【XX市电子政务智能助手】。请基于以下【参考案例】，回答市民的【最新提问】。

【参考案例】：
案例1（来源：{dept1}）：
问：{q1}
答：{a1}
...
案例5（来源：{dept5}）：
...

【最新提问】：{user_query}

【回答要求】：
1. **综合性**：如果参考案例涉及多个部门（如案例1是医保，案例3是税务），请综合说明办事流程，不要割裂回答。
2. **准确性**：严格基于参考案例中的政策依据，不要编造。如果没有相关信息，请回答“暂无相关案例”。
3. **结构化**：分点作答，语气亲切官方。
4. **溯源**：在回答末尾注明：“以上信息参考自 [XX局] 的历史回复”。
```

---

## 3. 下一步行动建议 (Action Plan)

为了让你能顺利在毕设中落地，建议按以下周计划执行：

### 第一周：环境搭建与数据清洗
1.  **硬件环境**：安装 PyTorch (CUDA 版本)，确保 `torch.cuda.is_available()` 返回 True。
2.  **申请 API**：去阿里云 DashScope 控制台申请 Qwen API Key，充值 20-50 元（足够毕设调试）。
3.  **数据清洗脚本**：
    *   编写 Python 脚本，读取你的 Excel/CSV。
    *   调用 Qwen-Turbo API 进行“数据增强”（生成模拟提问），将 4.5万条数据扩充并保存为 JSONL 格式。

### 第二周：构建索引 (Milvus + Embedding)
1.  **安装 Milvus Lite**：`pip install pymilvus milvus-lite`。
2.  **模型下载**：从 HuggingFace 或 ModelScope 下载 `BAAI/bge-m3` 模型文件到本地。
3.  **入库脚本**：
    *   加载 BGE-M3 模型。
    *   批量计算向量（Batch Size 设为 16 或 32，避免爆显存）。
    *   将向量和元数据插入 Milvus。

### 第三周：编写检索流水线
1.  **编写 `Retrieval` 类**：
    *   实现 `search_milvus()` 函数。
    *   实现 `rerank()` 函数（加载 `bge-reranker-v2-m3`）。
2.  **联调**：输入一个测试问题，打印出 Rerank 之后的 Top-5 案例，肉眼观察是否比你以前的聚类方法更准。

### 第四周：接入 Qwen 生成与 UI 开发
1.  **生成**：将 Top-5 案例填入 Prompt，调用 Qwen-Plus。
2.  **UI**：用 **Streamlit** 写一个简单的网页界面：
    *   输入框：用户提问。
    *   侧边栏：显示“正在路由到：XX局”、“已检索到 X 条案例”。
    *   主区域：显示最终回答和参考案例原文。

### 关键代码片段 (预告)
为了让你起步更顺，这里送你一段 **Reranker** 的核心代码：

```python
import torch
from FlagEmbedding import FlagReranker

# 1. 加载模型 (RTX 4050 显存优化版)
reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True) 

def compute_rerank_scores(user_query, candidate_docs):
    # 构建 Pairs: [[query, doc1], [query, doc2]...]
    pairs = [[user_query, doc] for doc in candidate_docs]
    
    # 计算得分
    scores = reranker.compute_score(pairs)
    
    # 结合索引排序
    ranked_results = sorted(zip(candidate_docs, scores), key=lambda x: x[1], reverse=True)
    
    return ranked_results[:5] # 返回 Top 5
```

这套方案不仅技术先进（涵盖了 RAG 领域 2024 年的主流技术），而且完全适配你的硬件和职业规划。祝你毕设顺利！