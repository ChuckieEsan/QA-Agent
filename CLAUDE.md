# GovPulse CLAUDE.md

## Project Overview

GovPulse 是一个基于 **LangGraph ReAct Agent** + **RAG** + **MCP (Model Context Protocol)** 架构的政务全流程自动化处理系统，作为 12345 政务服务热线的智能中枢。

核心功能：
- 智能解答市民咨询类诉求
- 精准分类与派单
- 基于 RAG Triad 的幻觉检测与双层校验
- MCP 协议对接下游政务微服务

## AI Coding Directives (必须严格遵守)

**本项目的核心开发准则，所有代码必须遵循：**

### 1. 强类型约束

- **类型注解**: 所有函数必须包含完整的 Type Hints
- **Pydantic 模型**: 核心业务数据传递必须使用 Pydantic (V2) 模型，**禁止使用裸字典 (`dict`)**
- **数据传输**: 跨层数据传递必须通过 Pydantic 模型

```python
# 正确: 使用 Pydantic 模型
class ChatRequest(BaseModel):
    query: str
    session_id: str

def process(request: ChatRequest) -> ChatResponse:
    ...

# 禁止: 裸字典
def process(data: dict) -> dict:
    ...
```

### 2 Prompt 外置原则

- **禁止硬编码**: 严禁将长篇 Prompt 字符串硬编码在 Python 文件中
- **单独存放**: 所有系统提示词必须作为 `.md` 或 `.txt` 文件存放在 `app/prompts/` 目录
- **按需读取**: 在代码中按需读取，保持提示词与代码解耦

```
app/prompts/
├── agent_system_prompt.md
├── classifier_prompt.md
└── validator_prompt.md
```

```python
# 正确: 从文件读取 Prompt
from pathlib import Path

def get_system_prompt() -> str:
    prompt_path = Path("app/prompts/agent_system_prompt.md")
    return prompt_path.read_text(encoding="utf-8")
```

### 3 日志优先原则

- **禁止 print**: 严禁在业务代码中使用 `print()` 函数
- **统一日志**: 使用 `app/utils/logger.py` 中的 `get_logger(__name__)` 进行日志记录
- **生产追踪**: 日志用于生产环境排查和 MQ 消费追踪

```python
# 正确
logger = get_logger(__name__)
logger.info(f"处理请求: {request_id}")

# 禁止
print(f"处理请求: {request_id}")
```

### 防御性编程

- **必含 try-except**: 调用 LLM API、数据库 (PostgreSQL/Milvus) 时必须包含 `try-except` 块
- **失败重试**: 在消费者端正确处理 Nack 或失败重试逻辑
- **优雅降级**: 外部服务不可用时应有降级策略

```python
# 正确的防御性编程
async def query_cases(query: str) -> List[Document]:
    try:
        return await dbClient.query(query)
    except ConnectionError as e:
        logger.error(f"数据库连接失败: {e}")
        raise
    except Exception as e:
        logger.error(f"查询失败: {e}", exc_info=True)
        return []  # 降级返回空列表
```

### 5. Emoji 使用规范

- **业务代码**: 禁止在业务代码中使用 Emoji 字符
- **前端 UI**: 在开发 Streamlit 前端时，可以适当使用 Emoji

```python
# 禁止: 业务代码中使用 Emoji
logger.info(f"✅ 请求处理成功")

# 正确: 简洁的日志
logger.info(f"请求处理成功")
```

### 6. LangChain/LangGraph 最佳实践

- **复用组件**: 尽可能少的重复造轮子，复用 LangChain 已有的组件
- **标准接口**: 继承 LangChain 标准接口（如 `BaseRetriever`, `RunnableSerializable`）
- **LCEL 链式**: 使用 LCEL 构建链式表达式

### 7. 重构原则

- **不向后兼容**: 如果发生重构，**不需要进行向后兼容**
- **直接重构**: 直接修改代码，不需要保留旧接口

### 8. 测试数据规范

- **只读生产库**: 禁止在测试用例中**写入**生产数据库，只允许读取
- **测试数据**: 使用独立的测试数据库或 mock 数据

## Coding Standards

本项目的代码审查规范，确保代码质量一致性和可维护性。

### 1. General Python Style

- **类型注解**: 必须使用类型注解，特别是函数参数和返回值
- **文档字符串**: 所有公开类、函数添加 docstring，使用中文描述
- **命名规范**:
  - 类名: `PascalCase` (如 `GovRequestClassifier`)
  - 函数/变量: `snake_case` (如 `create_gov_agent`)
  - 常量: `UPPER_SNAKE_CASE`
  - 私有成员: 前缀 `_` (如 `_internal_method`)
- **导入顺序**: 标准库 → 第三方库 → 项目内部模块，组间空行分隔

```python
# 正确的导入顺序
import os
import sys
from typing import List, Optional, Dict

from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from src.app.agents import ainvoke
from src.app.infra.utils.logger import get_logger
```

### 2. Pydantic V2 Usage

- **配置模型**: 使用 `BaseModel` + `Field` 定义配置类
- **字段验证**: 使用 `Field` 的 `ge`, `le`, `description` 参数
- **枚举类**: 使用 `Enum` + `property` 定义带中文描述的枚举

```python
# 正确的 Pydantic 用法
class PostgresDBConfig(BaseConfig):
    host: str = Field(default="localhost", description="PostgreSQL 主机")
    port: int = Field(default=5432, ge=1, le=65535, description="端口")
    enable_dynamic_field: bool = Field(default=True)

class GovRequestType(Enum):
    ADVICE = "advice"
    COMPLAINT = "complaint"

    @property
    def chinese(self):
        return {'ADVICE': '建议', 'COMPLAINT': '投诉'}[self.name]
```

### 3. LangGraph / LangChain Best Practices

- **RunnableSerializable**: LLM 服务类必须继承 `RunnableSerializable`，符合 LangChain 标准
- **异步优先**: 使用 `ainvoke` 而非 `invoke`，保持异步一致性
- **工具定义**: 使用 `@tool` 装饰器，定义 `args_schema` 明确参数类型
- **StateGraph 节点**: 节点函数必须标注类型 `Async def func(state: AgentState)`

```python
# 正确的 LangGraph 工具定义
class RetrieveCasesArgs(BaseModel):
    query: str = Field(..., description="查询关键词")
    top_k: int = Field(default=5, description="返回数量")

@tool("retrieve_cases_tool", args_schema=RetrieveCasesArgs)
def retrieve_cases_tool(query: str, top_k: int = 5) -> str:
    """【历史案例检索】..."""
    ...
```

### 4. Async/Await Patterns

- **异步函数**: 所有 I/O 操作使用异步函数
- **工具函数**: 如果涉及外部调用（如 LLM），工具必须是 `async def`
- **组合使用**: `await` 配合 `asyncio.run` 或在 async 上下文中调用

```python
# 正确的异步工具
@tool("validate_answer_tool")
async def validate_answer_tool(...) -> Union[str, Command]:
    result = await _validator.validate(...)
    return result

# 避免在 async 函数中使用阻塞调用
# 正确: result = await client.async_get(...)
# 错误: result = client.sync_get(...)
```

### 5. Dependency Injection

- **工厂函数**: 使用工厂函数创建复杂对象（如 `create_gov_request_classifier()`）
- **单例模式**: 配置类使用 `settings = Settings()` 单例
- **避免全局状态**: 组件通过参数注入，避免模块级可变状态

```python
# 工厂函数模式
def create_cases_retriever(top_k: int = 5) -> CasesVectorRetriever:
    """创建案例检索器"""
    embeddings = load_embeddings()
    return CasesVectorRetriever(embeddings=embeddings, top_k=top_k)

# 配置单例
from src.config import settings
db_host = settings.postgres_db.host
```

### 6. Testing Standards

- **测试文件命名**: `test_*.py`
- **测试函数命名**: `test_function_name()`
- **异步测试**: 使用 `pytest-asyncio`，添加 `@pytest.mark.asyncio` 装饰器
- **Mock 使用**: 使用 `pytest-mock`，但避免 mock 数据库/外部 API（集成测试需用真实环境）
- **断言**: 使用有意义的断言消息

```python
# 正确的测试写法
@pytest.mark.asyncio
async def test_classify_consult():
    classifier = create_gov_request_classifier()

    result = classifier.classify("咨询社保如何缴纳")

    assert result.request_type == GovRequestType.CONSULT
    assert result.request_department is not None
```

### 7. Code Review Checklist

提交代码前检查清单：

- [ ] **类型注解**: 函数参数和返回值是否都有类型注解？
- [ ] **文档字符串**: 公开类和函数是否有中文 docstring？
- [ ] **日志**: 关键操作是否有适当的日志记录？
- [ ] **异常处理**: 外部调用是否有 try-catch 并记录日志？
- [ ] **敏感信息**: 日志中是否暴露了 API Key、密码等敏感信息？
- [ ] **异步一致性**: 是否正确使用 async/await？
- [ ] **资源释放**: 是否有需要清理的资源（连接、文件等）？
- [ ] **测试覆盖**: 新功能是否添加了测试？
- [ ] **配置外置**: 硬编码的配置是否提取到 settings？
- [ ] **依赖注入**: 是否通过参数注入而非全局状态？

### 8. Security Guidelines

- **环境变量**: 敏感配置必须从环境变量读取，禁止硬编码
- **API Key**: 使用 `os.getenv()` 读取，禁止提交到版本控制
- **用户输入**: 使用 Pydantic 的 `Field(..., min_length, max_length)` 验证
- **SQL 注入**: 使用参数化查询，禁止字符串拼接 SQL
- **日志脱敏**: 输出日志前检查是否包含敏感信息


## Project Structure

```
QA-Agent/
├── src/
│   ├── app/
│   │   ├── prompts/              # 系统提示词 (外置)
│   │   │   └── *.md
│   │   ├── agents/           # ReAct Agent 编排层
│   │   │   ├── __init__.py   # ainvoke 入口函数
│   │   │   ├── react_agent.py # LangGraph StateGraph 定义
│   │   │   └── tools/        # Agent 工具集
│   │   │       ├── registry.py
│   │   │       ├── local_tools.py
│   │   │       └── mcp_tools.py
│   │   ├── api/              # FastAPI 服务层
│   │   │   ├── routes.py     # API 路由定义
│   │   │   └── server.py     # 服务启动入口
│   │   ├── components/       # 业务组件
│   │   │   ├── classifier/   # 意图分类器
│   │   │   ├── retriever/    # 检索器
│   │   │   ├── reranker/     # 重排模型
│   │   │   └── validator/    # 答案校验器
│   │   ├── infra/            # 基础设施层
│   │   │   ├── db/           # 数据库客户端
│   │   │   │   ├── postgres_db.py  # PostgreSQL + pgvector
│   │   │   │   └── milvus_db.py    # Milvus (可选)
│   │   │   ├── llm/          # LLM 服务
│   │   │   │   ├── providers/      # 多模型适配器
│   │   │   │   │   ├── deepseek_provider.py
│   │   │   │   │   ├── qwen_provider.py
│   │   │   │   │   └── ollama_provider.py
│   │   │   │   └── base_llm_service.py
│   │   │   ├── embedding/    # 向量化服务
│   │   │   └── reranker/     # 重排服务
│   │   └── ui/               # Streamlit UI (可选)
│   └── config/
│       └── setting.py        # Pydantic 配置管理
├── tests/                    # 测试目录
├── scripts/                  # 工具脚本
├── models/                   # 本地模型 (BGE-M3, BGE-Reranker)
└── data/                     # 数据目录
```

## Technology Stack

| Category | Technology |
|----------|------------|
| Agent 框架 | LangGraph, LangChain (LCEL) |
| LLM 支持 | DeepSeek, Qwen, Ollama |
| 微服务与协议 | MCP (Model Context Protocol), FastAPI |
| RAG 向量检索 | PostgreSQL + pgvector, psycopg2 |
| Embedding & Rerank | BGE-M3, BGE-Reranker-Base |
| 配置与工程化 | Pydantic V2, Python logging |
| 测试 | pytest, pytest-asyncio, pytest-cov |

## Architecture

### Core Workflow (ReAct Agent)

```
用户 query → Agent → [意图分类 → 知识检索 → 答案校验]
                    ↓
              校验通过 → 返回结果 (熔断)
                    ↓
              校验失败 → 继续迭代
                    ↓
              无法解答 → MCP 派单 (create_work_order)
```

### Key Innovation: Dynamic Circuit Breaker (熔断机制)

当 `validate_answer_tool` 校验通过时，直接返回 `Command(goto=END)` 携带最终回复，阻断后续 ReAct 循环，降低 Token 消耗 30%+。

参考：[src/app/agents/react_agent.py:72-75](src/app/agents/react_agent.py#L72-L75)

## Key Components

### 1. Agent Layer

- **入口**: `src/app/agents/__init__.py` 中的 `ainvoke()` 函数
- **核心**: [src/app/agents/react_agent.py](src/app/agents/react_agent.py) 定义 LangGraph StateGraph

### 2. Tools

- **意图分类**: `classify_gov_request_tool` - 提取诉求类型和管辖部门
- **案例检索**: `retrieve_cases_tool` - RAG 政策与历史案例检索
- **权责检索**: `retrieve_powers_tool` - 确认投诉归属部门
- **答案校验**: `validate_answer_tool` - RAG Triad 校验 (核心!)
- **工单创建**: `create_work_order` - MCP 远程工具调用

### 3. Database

- **PostgreSQL** (默认): `src/app/infra/db/postgres_db.py`
- **Milvus** (可选): `src/app/infra/db/milvus_db.py`
- 向量维度: 1024 (BGE-M3)
- 索引类型: HNSW

### 4. LLM Providers

- **Base**: `src/app/infra/llm/base_llm_service.py`
- **Providers**: DeepSeek, Qwen, Ollama (适配器模式)
- 配置加载: 环境变量 `DEEPSEEK_API_KEY`, `QWEN_API_KEY`, `OLLAMA_BASE_URL` 等

## API Reference

### POST /api/chat

政务问答主接口

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "灵活就业人员如何缴纳社保？", "session_id": "test-01"}'
```

**Request**:
- `query`: 用户查询 (必填, 1-1000 字符)
- `session_id`: 会话 ID (默认 "default")
- `top_k`: 检索结果数量 (默认 5, 1-20)

**Response**:
- `answer`: 生成的回答
- `classification`: 分类结果 `{type, request_department}`
- `sources`: 检索来源
- `quality_score`: 质量评分 (0-1)
- `work_order_id`: 工单 ID (如有)
- `timestamp`: 时间戳

### GET /api/health

健康检查

### GET /api/stats

获取系统统计信息

## Configuration

### Environment Variables

创建 `.env` 文件:

```bash
# LLM Providers
DEEPSEEK_API_KEY=your-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_GENERATION_MODEL=deepseek-chat

QWEN_API_KEY=your-key
QWEN_BASE_URL=https://dashscope.aliyuncs.com/api/v1

OLLAMA_BASE_URL=http://localhost:11434/v1

# Default Provider
LLM_DEFAULT_PROVIDER=deepseek

# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=root
POSTGRES_PASSWORD=root
POSTGRES_DATABASE=db
```

### Config Classes

配置位于 [src/config/setting.py](src/config/setting.py)，使用 Pydantic V2:
- `Settings` - 主配置类
- `LLMConfig` - 多提供商配置
- `PostgresDBConfig` - 数据库配置
- `RetrieverConfig` - 检索器配置
- `RetrieverConfig.threshold_strategy` - 阈值策略 (hybrid/fixed/dynamic/top_percentage)

## Testing

```bash
# 运行所有测试
uv run pytest

# 运行特定测试文件
uv run pytest tests/api/test_routes.py

# 运行带覆盖率
uv run pytest --cov=src --cov-report=term-missing

# 开发依赖
uv sync --group dev
```

测试配置: `pyproject.toml` 中的 `[tool.pytest.ini_options]`

## Development Guidelines

### Adding New LLM Provider

1. 在 `src/app/infra/llm/providers/` 创建新 provider 文件
2. 继承 `BaseLLMProvider` 实现接口
3. 在 `base_llm_service.py` 注册 provider
4. 在 `.env` 添加对应环境变量

### Adding New Tool

1. 在 `src/app/agents/tools/` 创建工具实现
2. 使用 `@tool` 装饰器定义工具函数
3. 在 `registry.py` 的 `get_all_tools()` 注册工具

### Database Schema

PostgreSQL 表:
- `gov_cases` - 问政案例
- `gov_powers` - 行政权力清单

向量字段: `embedding` (dimension: 1024)
索引: HNSW, cosine 距离

## Important Notes

1. **Python 版本**: >= 3.13
2. **包管理**: 使用 `uv` (`uv sync`, `uv run`)
3. **向量数据库选择**: 通过 `settings.db_type` 配置 (postgres/milvus)
4. **会话管理**: LangGraph MemorySaver 自动管理
5. **熔断机制**: 理解 `Command(goto=END)` 的工作原理是理解代码的关键

## Common Tasks

### 启动服务
```bash
uv run python -m src.app.api.server
```

### 调试配置
```bash
uv run python -m src.config.setting
```

### 数据导入
```bash
uv run python scripts/data/ingest.py
```

### 检索演示
```bash
uv run python scripts/demo/retrieve_demo.py
```

