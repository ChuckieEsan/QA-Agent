# GovPulse - 泸州市政务智能问答系统

## 项目概述

**GovPulse** 是一个基于检索增强生成（RAG）技术的泸州市政务智能问答系统，旨在为市民提供精准、可靠的政务政策咨询和民生问题解答服务。该系统结合了向量检索、多维度重排、意图分析和大语言模型生成等先进技术，构建了一个端云协同的智能问答解决方案。

**版本**: 0.1.0
**项目名称**: 泸州市政务智能问答系统 (Agentic RAG)
**技术栈**: Python 3.13 + PyTorch + Milvus + BGE-M3 + Qwen API

---

## 一、项目架构分析

### 1.1 架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                        应用层 (Application)                        │
│                    ┌─────────────────────┐                      │
│                    │    RagAgent         │  ← 主要交互入口        │
│                    └─────────────────────┘                      │
└─────────────────────────────────────────────────────────────────┘
                             │
┌─────────────────────────────┴─────────────────────────────────────┐
│                        组件层 (Components)                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │  Retriever  │  │  Generator  │  │ Classifier  │  │   Memory    │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                             │
┌─────────────────────────────┴─────────────────────────────────────┐
│                     基础设施层 (Infrastructure)                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │
│  │     LLM     │  │  Vector DB  │  │   Utils     │               │
│  │   Service   │  │   (Milvus)  │  │ (Logger...) │               │
│  └─────────────┘  └─────────────┘  └─────────────┘               │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 架构设计合理性评估

#### ✅ 优点

1. **分层架构清晰** - 采用应用层 → 组件层 → 基础设施层的三层架构，职责分明，便于维护和扩展

2. **面向接口编程** - 所有核心模块都定义了抽象基类（BaseRetriever, BaseGenerator, BaseClassifier, BaseMemory, BaseLLMService, BaseDBClient），符合依赖倒置原则，便于单元测试和模块替换

3. **单一职责原则** - 每个组件职责单一：
   - Retriever: 负责检索
   - Generator: 负责生成回答
   - Classifier: 负责问政类型分类
   - Memory: 负责对话历史管理
   - Agent: 负责协调各组件

4. **可扩展性强** - 通过抽象基类，可以轻松添加新的检索策略、生成模型、分类器等

5. **配置集中管理** - 使用 Pydantic 管理配置（Settings 类），支持环境变量覆盖，便于部署

6. **依赖注入设计** - Agent 构造时可以传入自定义组件实例，支持灵活的组合

7. **单例模式应用** - 对资源密集型组件（如 LLMService, Retriever）使用单例模式，避免重复初始化

#### ⚠️ 潜在问题与改进建议

1. **混合检索器部分功能未启用** - `HybridVectorRetriever` 中的阈值筛选和重排功能被注释（TODO 标记），需要完善

2. **缺少API层** - `src/app/api/` 目录为空，如果计划提供Web API，需要补充 FastAPI/Flask 接口

3. **测试覆盖率不足** -
   - `tests/services/test_retriever.py` 为空
   - 缺少集成测试
   - 建议补充：`tests/agents/`, `tests/components/` 等测试目录

### 1.3 架构建议

1. **完善检索器功能** - 启用混合重排和阈值筛选功能，提升检索质量

2. **补充测试** - 增加集成测试和端到端测试

3. **文档完善** - 补充README和API文档

---

## 二、技术架构详解

### 2.1 核心组件

#### 2.1.1 Agent 层 (`src/app/agents/`)

**RagAgent** - 核心协调者
- 职责：协调检索、分类、生成、记忆四大组件，实现完整RAG流程
- 输入：用户查询、对话历史
- 输出：回答、分类结果、检索来源、质量评分
- 特点：
  - 支持依赖注入，可传入自定义组件
  - 实现 Agent 模式，统一处理流程
  - 提供同步和异步两种接口

#### 2.1.2 组件层 (`src/app/components/`)

**1. Retriever 组件** (`src/app/components/retrievers/`)
- `BaseRetriever`: 抽象基类，定义检索接口
- `HybridVectorRetriever`: 混合向量检索器实现
  - 支持向量检索 + 多维度重排（相似度、时效性、权威性、长度）
  - 支持缓存机制
  - 单例模式

**2. Generator 组件** (`src/app/components/generators/`)
- `BaseGenerator`: 抽象基类，定义生成接口
- `LLMGenerator`: LLM生成器实现
  - 封装 LLM Service
  - 支持同步/流式生成
  - 支持生成质量验证

**3. Classifier 组件** (`src/app/components/classifier/`)
- `BaseClassifier`: 抽象基类，定义分类接口
- `GovClassifier`: 政务问政分类器
  - 基于LLM的问政请求分类（建议/投诉/求助/咨询）
  - 支持批量分类

**4. Memory 组件** (`src/app/components/memory/`)
- `BaseMemory`: 抽象基类，定义记忆接口
- `ConversationMemory`: 对话记忆实现
  - 内存存储对话历史
  - 支持保存/加载到文件
  - 提供统计信息

#### 2.1.3 基础设施层 (`src/app/infra/`)

**1. LLM 服务** (`src/app/infra/llm/`)
- `BaseLLMService`: 抽象基类
- `LLMService`: Qwen API 封装
  - 支持意图分析（Agent决策）
  - 支持回答生成
  - 支持回答质量校验
  - 单例模式

**2. 向量数据库** (`src/app/infra/db/`)
- `BaseDBClient`: 抽象基类
- `MilvusDBClient`: Milvus Lite 客户端实现
  - 封装CRUD操作
  - 支持集合管理
  - 单例模式

**3. 工具类** (`src/app/infra/utils/`)
- `logger.py`: 统一日志管理（彩色输出、轮转文件）
- `data_utils.py`: 数据工具（生成文档ID、文本清洗）
- `system_utils.py`: 系统工具（设备检测）

### 2.2 配置系统 (`src/config/`)

**Settings** - 集中式配置管理
- 使用 Pydantic BaseModel 进行类型检查
- 支持环境变量覆盖（.env文件）
- 配置模块：
  - `PathConfig`: 路径配置
  - `ModelConfig`: 模型配置（Embedding、重排）
  - `MilvusDBConfig`: 向量数据库配置
  - `RetrieverConfig`: 检索器配置（阈值、重排权重）
  - `LLMConfig`: LLM配置（API Key、生成参数）
  - `LoggingConfig`: 日志配置
  - `PerformanceConfig`: 性能配置

### 2.3 数据流程

```
1. 用户提问
   ↓
2. RagAgent 接收查询
   ↓
3. 分类器分类问政类型 (GovClassifier)
   ↓
4. 检索器检索相关案例 (HybridVectorRetriever)
   ├── 向量检索（BGE-M3）
   ├── 阈值筛选（TODO: 未启用）
   ├── 多维度重排（TODO: 未启用）
   └── 缓存查询
   ↓
5. 生成器生成回答 (LLMGenerator)
   ├── 构建Prompt（包含检索结果）
   ├── 调用LLM Service
   └── 质量校验
   ↓
6. 保存到记忆 (ConversationMemory)
   ↓
7. 返回结果（回答 + 来源 + 质量评分）
```

---

## 三、模块详细说明

### 3.1 检索器模块 (`src/app/components/retrievers/`)

#### HybridVectorRetriever

**核心功能**：
1. **向量检索** - 使用 BGE-M3 模型将查询向量化，在Milvus中检索相似案例
2. **缓存机制** - 对相同查询进行缓存，提升性能
3. **混合重排** - 基于多维度评分进行结果重排（相似度/时效性/权威性/长度）
4. **阈值筛选** - 基于相似度阈值过滤低质量结果

**配置参数**：
```python
{
    "top_k": 5,                    # 默认返回结果数
    "cache_enabled": True,         # 是否启用缓存
    "cache_ttl": 300,              # 缓存过期时间（秒）
    "min_similarity": 0.65,        # 基础相似度阈值
    "min_results": 3,              # 最小返回结果数
    "max_results": 10,             # 最大返回结果数
    "weight_similarity": 0.8,      # 相似度权重
    "weight_recency": 0.7,         # 时效性权重
    "weight_authority": 0.2,       # 部门权威性权重
    "weight_length": 0.1           # 内容长度权重
}
```

**当前状态**：阈值筛选和重排功能已实现但被注释（TODO标记），需启用

### 3.2 LLM 服务模块 (`src/app/infra/llm/`)

#### LLMService

**核心功能**：
1. **意图分析** - 分析用户查询意图，生成检索决策（AgentDecision）
2. **回答生成** - 基于检索结果生成专业回答
3. **质量校验** - 对生成回答进行质量评估

**Agent决策类型**：
- `direct_answer`: 无需检索，直接回答
- `need_retrieval`: 需要检索后回答
- `multi_retrieval`: 需要多策略检索
- `cannot_answer`: 无法回答

**检索策略**：
- `hybrid`: 混合向量检索（默认）
- `keyword`: 关键词检索
- `semantic_only`: 纯语义检索
- `cross_dept`: 跨部门检索

### 3.3 数据库模块 (`src/app/infra/db/`)

#### MilvusDBClient

**核心功能**：
1. **集合管理** - 创建/删除/描述集合
2. **数据操作** - 插入/搜索/查询/删除/更新
3. **统计信息** - 获取集合统计信息

**集合Schema**：
```python
{
    "vector": FLOAT_VECTOR(1024),   # BGE-M3向量
    "text": VARCHAR(65535),         # RAG上下文
    "department": VARCHAR(255),     # 部门名称
    "metadata": JSON                # 元数据（title, question, answer, time, url, doc_id）
}
```

---

## 四、项目结构说明

### 4.1 目录结构

```
QA-Agent/
├── src/                        # 源代码目录
│   ├── app/                    # 应用代码
│   │   ├── agents/             # Agent层（RagAgent）
│   │   ├── components/         # 业务组件
│   │   │   ├── retrievers/     # 检索器
│   │   │   ├── generators/     # 生成器
│   │   │   ├── classifier/     # 分类器
│   │   │   └── memory/         # 记忆组件
│   │   ├── infra/              # 基础设施
│   │   │   ├── llm/            # LLM服务
│   │   │   ├── db/             # 数据库客户端
│   │   │   └── utils/          # 工具类
│   │   ├── api/                # API接口（待实现）
│   │   └── __init__.py         # 公共API导出
│   └── config/                 # 配置管理
│       └── setting.py          # 配置类
│
├── tests/                      # 测试代码
│   ├── services/               # 服务层测试
│   ├── utils/                  # 工具类测试
│   ├── conftest.py             # 测试配置
│   └── __init__.py
│
├── scripts/                    # 脚本目录
│   ├── data/                   # 数据处理脚本
│   │   ├── crawl.py            # 爬虫
│   │   ├── ingest.py           # 数据入库
│   │   └── gen_test_data.py    # 生成测试数据
│   ├── demo/                   # 演示脚本
│   │   ├── chat_demo.py        # 聊天演示
│   │   ├── retrieve_demo.py    # 检索演示
│   │   └── db_viewer.py        # 数据库查看器（Streamlit）
│   └── evaluation/             # 评估脚本
│       └── retrieve_baseline.py # 基线评估
│
├── data/                       # 数据目录
│   ├── raw/                    # 原始数据
│   ├── processed/              # 处理后数据
│   ├── milvus_db/              # Milvus数据库文件
│   └── raw_data.db             # SQLite原始数据
│
├── models/                     # 模型目录
│   ├── bge-m3/                 # Embedding模型
│   └── bge-reranker-base/      # 重排模型
│
├── logs/                       # 日志目录
├── docs/                       # 文档目录
├── .env                        # 环境变量
├── pyproject.toml              # 项目配置
└── uv.lock                     # 依赖锁定
```

### 4.2 依赖关系图

```
src.app.agents.RagAgent
    ├── src.app.components.retrievers.HybridVectorRetriever
    │   ├── src.app.infra.db.MilvusDBClient
    │   ├── sentence_transformers.SentenceTransformer
    │   └── src.config.setting
    ├── src.app.components.generators.LLMGenerator
    │   └── src.app.infra.llm.LLMService
    │       ├── dashscope.Generation
    │       └── src.config.setting
    ├── src.app.components.classifier.GovClassifier
    │   └── src.app.infra.llm.LLMService
    └── src.app.components.memory.ConversationMemory
```

---

## 五、核心使用示例

### 5.1 快速开始

```python
import asyncio
from src import RagAgent, query_agentic_rag

async def main():
    # 方式1: 使用工具函数
    result = await query_agentic_rag(
        query="2024年泸州市雨露计划补贴标准",
        history=[]
    )

    print(f"回答: {result['answer']}")
    print(f"分类: {result['classification']}")
    print(f"来源: {len(result['sources'])}个案例")
    print(f"质量评分: {result['quality_check']['overall_score']}")

    # 方式2: 使用Agent实例
    agent = RagAgent()
    result = await agent.process("如何办理社保？")

    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

### 5.2 检索器使用

```python
from src import get_retriever_instance, retrieve_with_details

# 方式1: 获取单例实例
retriever = get_retriever_instance()
context, results, metadata = retriever.retrieve("雨露计划什么时候发放？")

# 方式2: 快捷函数
result = retrieve_with_details("雨露计划", top_k=5)
print(result["sources"])
```

### 5.3 LLM服务使用

```python
from src import get_llm_service

llm = get_llm_service()
response = await llm.generate_response(
    query="用户问题",
    context="检索到的上下文",
    history=[{"role": "user", "content": "之前的问题"}]
)

print(response["answer"])
print(response["usage"])  # Token使用统计
```

---

## 六、配置说明

### 6.1 环境变量配置 (`.env`)

```bash
# DashScope API Key
DASHSCOPE_API_KEY=sk-xxxxxx

# 可选：调试模式
DEBUG=true
```

### 6.2 配置类使用

```python
from src import settings

# 访问配置
print(settings.project_name)  # GovPulse
print(settings.version)       # 1.0.0
print(settings.models.embedding_model)  # bge-m3
print(settings.llm.model_name)          # qwen-max

# 访问子配置
print(settings.paths.data_dir)
print(settings.vectordb.collection_name)
print(settings.retriever.base_threshold)
```

---

## 七、开发与测试

### 7.1 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/utils/test_data.py -v

# 生成覆盖率报告
pytest --cov=src --cov-report=html
```

### 7.2 数据处理流程

1. **数据爬取** (`scripts/data/crawl.py`)
   ```bash
   python scripts/data/crawl.py
   ```

2. **数据入库** (`scripts/data/ingest.py`)
   ```bash
   python scripts/data/ingest.py
   ```

3. **生成测试数据** (`scripts/data/gen_test_data.py`)
   ```bash
   python scripts/data/gen_test_data.py
   ```

### 7.3 演示脚本

1. **聊天演示** (`scripts/demo/chat_demo.py`)
   ```bash
   python scripts/demo/chat_demo.py
   ```

2. **检索演示** (`scripts/demo/retrieve_demo.py`)
   ```bash
   python scripts/demo/retrieve_demo.py
   ```

3. **数据库查看器** (`scripts/demo/db_viewer.py`)
   ```bash
   streamlit run scripts/demo/db_viewer.py
   ```

---

## 八、部署与运行

### 8.1 环境要求

- Python >= 3.13
- GPU (推荐): RTX 4050 或更高（用于Embedding模型）
- RAM: >= 8GB
- 磁盘: >= 10GB（模型+数据）

### 8.2 安装依赖

```bash
# 使用 uv (推荐)
uv sync

# 或使用 pip
pip install -e .
pip install -e ".[dev]"
```

### 8.3 模型准备

1. 下载 BGE-M3 模型到 `models/bge-m3/`
2. （可选）下载 BGE-Reranker 模型到 `models/bge-reranker-base/`

### 8.4 运行系统

```bash
# 1. 爬取数据（首次运行）
python scripts/data/crawl.py

# 2. 数据入库
python scripts/data/ingest.py

# 3. 运行检索演示
python scripts/demo/retrieve_demo.py

# 4. 运行聊天演示
python scripts/demo/chat_demo.py
```

---

## 九、架构设计决策记录

### 9.1 为什么选择三层架构？

- **清晰的职责分离**：应用层关注业务逻辑，组件层关注可复用模块，基础设施层关注外部依赖
- **便于测试**：每层可以独立测试
- **便于维护**：修改一层不会影响其他层

### 9.2 为什么使用抽象基类？

- **依赖倒置**：高层模块不依赖低层模块，都依赖抽象
- **易于替换**：可以轻松替换实现（如换用Elasticsearch替代Milvus）
- **便于测试**：可以使用Mock对象进行单元测试

### 9.3 为什么使用单例模式？

- **资源优化**：避免重复加载模型和数据库连接
- **全局一致**：确保所有地方使用同一个实例
- **简化使用**：通过工厂函数获取实例

### 9.4 为什么使用Milvus Lite而非完整版？

- **轻量级**：适合单机部署和开发测试
- **文件存储**：数据持久化到本地文件
- **简单易用**：无需复杂的集群配置

---

## 十、未来规划

### 10.1 短期目标

1. ✅ 启用混合重排和阈值筛选功能
2. ✅ 补充API层（FastAPI）
3. ✅ 增加集成测试覆盖率
4. ⏳ 实现Redis缓存
5. ⏳ 实现Reranker重排模型

### 10.2 中期目标

1. 实现Web管理后台
2. 支持多轮对话
3. 实现用户反馈机制
4. 优化检索性能（HNSW索引）

### 10.3 长期目标

1. 支持多语言
2. 实现多模态检索（图片、文档）
3. 集成更多政务数据源
4. 实现分布式部署

---

## 十一、贡献指南

### 11.1 代码风格

- 遵循 PEP 8 规范
- 使用类型注解
- 编写清晰的文档字符串
- 提交前运行测试

### 11.2 提交流程

1. 创建特性分支：`git checkout -b feat/xxx`
2. 开发并测试
3. 提交代码：`git commit -m "feat(scope): description"`
4. 推送并创建PR

### 11.3 Commit 规范

- `feat`: 新功能
- `fix`: Bug修复
- `docs`: 文档更新
- `style`: 代码格式
- `refactor`: 重构
- `test`: 测试相关
- `chore`: 构建/工具

---

## 十二、常见问题

### Q1: 如何修改检索参数？

```python
from src import settings

# 临时修改
settings.retriever.base_threshold = 0.7
settings.retriever.max_results = 15

# 或在 .env 文件中设置
# RETRIEVER__BASE_THRESHOLD=0.7
# RETRIEVER__MAX_RESULTS=15
```

### Q2: 如何添加自定义组件？

```python
from src.app.components.retrievers import BaseRetriever

class CustomRetriever(BaseRetriever):
    def initialize(self):
        # 初始化资源
        pass

    def retrieve(self, query, top_k=None, **kwargs):
        # 实现检索逻辑
        return context, results, metadata

    def retrieve_with_details(self, query, top_k=None, **kwargs):
        # 实现详细检索逻辑
        return details

# 使用自定义组件
agent = RagAgent(retriever=CustomRetriever())
```

### Q3: 如何调试日志？

```python
from src import settings

# 临时修改日志级别
settings.logging.level = "DEBUG"

# 或使用日志上下文管理器
from src.app.infra.utils.logger import LoggingContext

with LoggingContext("DEBUG"):
    # 调试代码
    pass
```

---

## 十三、致谢

- **模型提供**: BAAI (BGE-M3), Alibaba (Qwen)
- **数据库**: Milvus
- **框架**: Python, PyTorch, FastAPI

---

## 十四、许可证

本项目采用 Apache 2.0 许可证。

---

**文档版本**: v1.0
**最后更新**: 2026-02-19
**维护者**: GovPulse Team
