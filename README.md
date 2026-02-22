# GovPulse - 泸州市政务智能问答系统

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

**ReAct Agent RAG** 版本

</div>

**GovPulse** 是一个基于检索增强生成（RAG）技术的泸州市政务智能问答系统，旨在为市民提供精准、可靠的政务政策咨询和民生问题解答服务。该系统结合了向量检索、多维度重排、意图分析和大语言模型生成等先进技术，构建了一个端云协同的智能问答解决方案。

**版本**: 0.2.0
**项目名称**: 泸州市政务智能问答系统 (ReAct Agent RAG)
**技术栈**: Python 3.13 + PyTorch + Milvus + BGE-M3 + Qwen API

---

---

## 一、项目架构分析

### 1.1 架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                        应用层 (Application)                        │
│                    ┌─────────────────────┐                      │
│                    │   ReactAgent        │  ← 纯 ReAct Agent入口  │
│                    └─────────────────────┘                      │
└─────────────────────────────────────────────────────────────────┘
                             │
┌─────────────────────────────┴─────────────────────────────────────┐
│                        组件层 (Components)                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │  Retriever  │  │  Generator  │  │ Classifier  │  │ Validator   │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                             │
┌─────────────────────────────┴─────────────────────────────────────┐
│                    基础设施层 (Infrastructure)                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │
│  │     LLM     │  │  Vector DB  │  │   Tools     │               │
│  │   Service   │  │   (Milvus)  │  │ (Registry)  │               │
│  └─────────────┘  └─────────────┘  └─────────────┘               │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 架构设计合理性评估

#### ✅ 优点

1. **纯 ReAct 范式** - 采用 Thought-Action-Observation 循环，LLM 自主决定推理步骤，更灵活

2. **工具注册表模式** - 使用 `@ToolRegistry.register()` 装饰器模式，便于扩展和管理工具

3. **分层架构清晰** - 采用应用层 → 组件层 → 基础设施层的三层架构，职责分明，便于维护和扩展

4. **面向接口编程** - 所有核心模块都定义了抽象基类，符合依赖倒置原则，便于单元测试和模块替换

5. **单一职责原则** - 每个组件职责单一：
   - Retriever: 负责检索
   - Generator: 负责生成回答
   - Classifier: 负责问政类型分类
   - Validator: 负责回答质量验证
   - Agent: 负责协调各工具

6. **可扩展性强** - 通过抽象基类和工具注册表，可以轻松添加新的工具或组件

7. **配置集中管理** - 使用 Pydantic 管理配置（Settings 类），支持环境变量覆盖，便于部署

8. **单例模式应用** - 对资源密集型组件（如 LLMService, Retriever）使用单例模式，避免重复初始化

#### ⚠️ 潜在问题与改进建议

1. **API 层待完善** - `src/app/api/` 目录已存在基础路由，可进一步完善文档和认证

2. **测试覆盖率待提升** - `tests/agents/test_react_agent.py` 已实现，可继续增加边缘 case 测试

### 1.3 架构建议

1. **启用混合重排** - 检索器已实现阈值筛选和重排功能，可启用以提升检索质量
2. **完善测试** - 增加集成测试和端到端测试
3. **API 文档** - 补充FastAPI 自动生成的 Swagger 文档说明

---

## 二、技术架构详解

### 2.1 核心组件

#### 2.1.1 Agent 层 (`src/app/agents/`)

**ReactAgent** - 纯 ReAct Agent
- 职责：通过 Thought-Action-Observation 循环自主决定推理步骤
- 输入：用户查询、工具集
- 输出：回答、推理步骤历史、检索来源
- 特点：
  - 纯 ReAct 范式：LLM 自主决定每步思考和工具调用
  - 工具注册表模式：装饰器 `@ToolRegistry.register()` 自动注册工具
  - 支持多步推理，最大步数可配置

**工具注册表 (ToolRegistry)**
- 提供工具的注册、查找和实例化功能
- 支持自定义工具扩展
- 内置工具：
  - `retrieve`: 检索相关案例和政策文档
  - `generate`: 生成回答文本
  - `classify`: 分类问政类型（建议/投诉/求助/咨询）
  - `validate`: 验证回答质量

**工具协议 (BaseTool)**
```python
class BaseTool(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...
    async def execute(self, **kwargs) -> dict: ...
```

#### 2.1.2 组件层 (`src/app/components/`)

**1. Retriever 组件** (`src/app/components/retrievers/`)
- `BaseRetriever`: 抽象基类，定义检索接口
- `HybridVectorRetriever`: 混合向量检索器实现
  - 支持向量检索 + 多维度重排（相似度、时效性、长度）
  - 支持缓存机制
  - 单例模式

**2. Generator 组件** (`src/app/components/generators/`)
- `BaseGenerator`: 抽象基类，定义生成接口
- `LLMGenerator`: LLM生成器实现
  - 封装 LLM Service
  - 支持同步/流式生成

**3. Classifier 组件** (`src/app/components/classifier/`)
- `BaseClassifier`: 抽象基类，定义分类接口
- `GovClassifier`: 政务问政分类器
  - 基于LLM的问政请求分类（建议/投诉/求助/咨询）
  - 支持批量分类

**4. Validator 组件** (`src/app/components/quality/`)
- `BaseValidator`: 抽象基类，定义验证接口
- `AnswerValidator`: 回答质量验证器
  - 评估相关性、完整性、准确性
  - 返回综合评分和反馈

#### 2.1.3 基础设施层 (`src/app/infra/`)

**1. LLM 服务** (`src/app/infra/llm/`)
- `BaseLLMService`: 抽象基类
- `multi_model_service.py`: 多模型 LLM 服务管理器
  - `get_optimizer_llm_service()`: 获取优化模型（用于思考和动作生成）
  - `get_heavy_llm_service()`: 获取主模型（用于最终答案生成）
  - 支持意图分析
  - 支持回答生成
  - 支持回答质量校验

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

### 2.3 数据流程 (ReAct Agent)

```
┌─────────────────────────────────────────────────────────────┐
│                    ReAct 推理循环                            │
│  Thought → Action → Observation → Thought → ...            │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│ 1.Thought: LLM 分析当前状态并生成思考                       │
│ 2.Action: 选择并执行工具 (retrieve/generate/classify/...)  │
│ 3.Observation: 获取工具执行结果                             │
│ 4.循环直到生成最终答案或达到最大步数                        │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│ 5.使用主模型生成最终答案                                     │
│ 6.返回结果（回答 + 步骤历史 + 来源）                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、模块详细说明

### 3.1 检索器模块 (`src/app/components/retrievers/`)

#### HybridVectorRetriever

**核心功能**：
1. **向量检索** - 使用 BGE-M3 模型将查询向量化，在Milvus中检索相似案例
2. **缓存机制** - 对相同查询进行缓存，提升性能
3. **混合重排** - 基于多维度评分进行结果重排（相似度/时效性/长度）
4. **阈值筛选** - 基于相似度阈值过滤低质量结果

**配置参数**：
```python
{
    "top_k": 5,                    # 默认返回结果数
    "cache_enabled": True,         # 是否启用缓存
    "cache_ttl": 300,              # 缓存过期时间（秒）
    "min_similarity": 0.5,         # 基础相似度阈值
    "min_results": 3,              # 最小返回结果数
    "max_results": 10,             # 最大返回结果数
    "weight_similarity": 0.6,      # 相似度权重
    "weight_recency": 0.3,         # 时效性权重
    "weight_length": 0.1           # 内容长度权重
}
```

**当前状态**：所有功能已实现并启用

### 3.2 工具注册表 (`src/app/agents/tools/`)

#### ToolRegistry

**核心功能**：
1. **装饰器注册** - 通过 `@ToolRegistry.register()` 自动注册工具
2. **实例管理** - 工具类注册时自动创建单例实例
3. **动态查找** - 通过名称获取工具实例

**使用示例**：
```python
from src.app.agents.tools.registry import ToolRegistry

@ToolRegistry.register()
class MyTool(BaseTool):
    name = "my_tool"
    description = "我的工具"

    async def execute(self, **kwargs) -> dict:
        return {"result": "xxx"}

# 获取工具实例
tool = ToolRegistry.get_instance("my_tool")
```

**内置工具**：
- `RetrievalTool`: 检索相关案例和政策文档
- `GenerationTool`: 生成回答文本
- `ClassificationTool`: 分类问政类型
- `ValidationTool`: 验证回答质量

### 3.3 LLM 服务模块 (`src/app/infra/llm/`)

#### multi_model_service

**核心功能**：
1. **多模型管理** - 根据任务类型选择合适的模型
2. **意图分析** - 分析用户查询意图，生成检索决策（AgentDecision）
3. **回答生成** - 基于检索结果生成专业回答
4. **质量校验** - 对生成回答进行质量评估

**模型服务**：
- `get_optimizer_llm_service()`: 获取优化模型（用于思考和动作生成）
- `get_heavy_llm_service()`: 获取主模型（用于最终答案生成）
- `get_light_llm_service()`: 获取轻量模型（用于快速响应）

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

### 3.4 数据库模块 (`src/app/infra/db/`)

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
│   │   ├── agents/             # Agent层（ReactAgent + Tools）
│   │   │   ├── tools/          # 工具模块
│   │   │   │   ├── registry.py     # 工具注册表
│   │   │   │   ├── base_tool.py    # 工具基类
│   │   │   │   ├── retrieval_tool.py
│   │   │   │   ├── generation_tool.py
│   │   │   │   ├── classification_tool.py
│   │   │   │   └── validation_tool.py
│   │   │   ├── models/         # 数据模型
│   │   │   │   └── agent_decision.py
│   │   │   ├── base_agent.py   # Agent 基类
│   │   │   ├── react_agent.py  # ReAct Agent 实现
│   │   │   └── __init__.py
│   │   ├── components/         # 业务组件
│   │   │   ├── retrievers/     # 检索器
│   │   │   ├── generators/     # 生成器
│   │   │   ├── classifier/     # 分类器
│   │   │   ├── memory/         # 记忆组件
│   │   │   └── quality/        # 质量验证
│   │   ├── infra/              # 基础设施
│   │   │   ├── llm/            # LLM服务
│   │   │   │   ├── multi_model_service.py
│   │   │   │   ├── base_llm_service.py
│   │   │   │   └── schema.py
│   │   │   ├── db/             # 数据库客户端
│   │   │   │   ├── milvus_db.py
│   │   │   │   └── base_db.py
│   │   │   └── utils/          # 工具类
│   │   │       ├── logger.py
│   │   │       ├── data_utils.py
│   │   │       └── system_utils.py
│   │   ├── api/                # API接口
│   │   │   ├── app.py          # FastAPI app
│   │   │   └── routes.py       # API路由
│   │   └── __init__.py         # 公共API导出
│   └── config/                 # 配置管理
│       └── setting.py          # 配置类
│
├── tests/                      # 测试代码
│   ├── agents/                 # Agent测试
│   │   └── test_react_agent.py
│   ├── components/             # 组件测试
│   │   └── retrievers/         # 检索器测试
│   ├── infra/                  # 基础设施测试
│   │   └── utils/              # 工具类测试
│   ├── config/                 # 配置测试
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
src.app.agents.ReactAgent
    ├── src.app.agents.tools.ToolRegistry
    │   └── src.app.agents.tools.BaseTool
    ├── src.app.components.retrievers.HybridVectorRetriever
    │   ├── src.app.infra.db.MilvusDBClient
    │   ├── sentence_transformers.SentenceTransformer
    │   └── src.config.setting
    ├── src.app.components.generators.LLMGenerator
    │   └── src.app.infra.llm.multi_model_service
    │       ├── dashscope.Generation
    │       └── src.config.setting
    ├── src.app.components.classifier.GovClassifier
    │   └── src.app.infra.llm.multi_model_service
    └── src.app.components.quality.AnswerValidator
```

---

## 五、核心使用示例

### 5.1 快速开始 (纯 ReAct Agent)

```python
import asyncio
from src import ReactAgent, ToolRegistry

async def main():
    # 创建工具集
    tools = {
        "retrieve": ToolRegistry.get_instance("retrieve"),
        "generate": ToolRegistry.get_instance("generate"),
        "classify": ToolRegistry.get_instance("classify"),
        "validate": ToolRegistry.get_instance("validate"),
    }

    # 创建 ReactAgent
    agent = ReactAgent(tools, max_steps=5)

    # 方式1: 直接使用 Agent
    result = await agent.process("2024年泸州市雨露计划补贴标准")

    print(f"回答: {result['answer']}")
    print(f"推理步数: {result['steps_count']}")
    print(f"来源: {len(result['sources'])} 个案例")
    print(f"步骤历史: {len(result['steps_history'])} 步")

    # 打印推理步骤
    for step in result['steps_history']:
        print(f"  Step {step['step_number']}: {step['action']}")

    # 方式2: 使用 API 接口（见 5.5）

if __name__ == "__main__":
    asyncio.run(main())
```

### 5.2 自定义工具

```python
from src.app.agents.tools.registry import ToolRegistry
from src.app.agents.tools.base_tool import BaseTool

# 注册自定义工具
@ToolRegistry.register()
class CalculatorTool(BaseTool):
    name = "calculator"
    description = "计算数学表达式"

    async def execute(self, expression: str = "") -> dict:
        try:
            result = eval(expression)
            return {"result": result}
        except:
            return {"result": "计算失败"}

# 使用自定义工具
tools["calculator"] = ToolRegistry.get_instance("calculator")
agent = ReactAgent(tools, max_steps=5)
```

### 5.3 检索器使用

```python
from src.app.components.retrievers import HybridVectorRetriever

# 使用单例实例
retriever = HybridVectorRetriever()
context, results, metadata = retriever.retrieve("雨露计划什么时候发放？", top_k=5)

# 查看检索结果
print(f"检索到 {len(results)} 个结果")
print(f"平均相似度: {metadata['avg_similarity']:.2f}")
```

### 5.4 LLM服务使用

```python
from src import get_optimizer_llm_service, get_heavy_llm_service

# 优化模型（用于思考和动作生成）
optimizer_llm = get_optimizer_llm_service()

# 主模型（用于最终答案生成）
heavy_llm = get_heavy_llm_service()
```

### 5.5 API 接口使用

```python
# 启动 API 服务
# uvicorn src.app.api.app:app --reload --host 0.0.0.0 --port 8000

# 聊天接口
curl -X POST http://localhost:8000/api/chat \
    -H "Content-Type: application/json" \
    -d '{"query": "雨露计划什么时候发放？", "history": [], "top_k": 5}'

# 健康检查
curl http://localhost:8000/api/health

# 统计信息
curl http://localhost:8000/api/stats
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

### 9.2 为什么使用纯 ReAct 范式？

- **自主推理**：LLM 自主决定每步思考和工具调用，更灵活
- **可解释性强**：.clear 的 Thought-Action-Observation 循环
- **易于扩展**：添加新工具即可扩展能力，无需修改 Agent 逻辑

### 9.3 为什么使用工具注册表模式？

- **装饰器注册**：通过 `@ToolRegistry.register()` 自动注册和实例化
- **便于管理**：集中管理所有工具
- **易于扩展**：自定义工具只需继承 BaseTool 并注册

### 9.4 为什么使用抽象基类？

- **依赖倒置**：高层模块不依赖低层模块，都依赖抽象
- **易于替换**：可以轻松替换实现（如换用Elasticsearch替代Milvus）
- **便于测试**：可以使用Mock对象进行单元测试

### 9.5 为什么使用单例模式？

- **资源优化**：避免重复加载模型和数据库连接
- **全局一致**：确保所有地方使用同一个实例
- **简化使用**：通过工厂函数获取实例

### 9.6 为什么使用Milvus Lite而非完整版？

- **轻量级**：适合单机部署和开发测试
- **文件存储**：数据持久化到本地文件
- **简单易用**：无需复杂的集群配置

---

## 十、未来规划

### 10.1 短期目标

1. ✅ 实现纯 ReAct Agent 架构
2. ✅ 添加工具注册表模式
3. ✅ 完善 API 层（FastAPI）
4. ✅ 增加集成测试覆盖率
5. ⏳ 实现Redis缓存

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

### Q2: 如何添加自定义工具？

```python
from src.app.agents.tools.registry import ToolRegistry
from src.app.agents.tools.base_tool import BaseTool

# 注册自定义工具
@ToolRegistry.register()
class CustomTool(BaseTool):
    name = "custom_tool"
    description = "我的自定义工具"

    async def execute(self, **kwargs) -> dict:
        return {"result": "xxx"}

# 使用自定义工具
tool = ToolRegistry.get_instance("custom_tool")
```

### Q3: ReactAgent 和 RagAgent 有什么区别？

| 特性 | RagAgent | ReactAgent |
|------|----------|------------|
| 推理方式 | 预定义流程 | LLM 自主推理 |
| 工具调用 | 手动编排 | 装饰器注册 |
| 灵活性 | 固定流程 | 动态决策 |
| 可扩展性 | 需修改代码 | 添加工具即可 |

### Q4: 如何调试日志？

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

**文档版本**: v0.2.0 (ReAct Agent RAG)
**最后更新**: 2026-02-22
**维护者**: GovPulse Team
