# GovPulse 测试指南

## 测试结构

本项目采用标准的 Python 测试结构，测试文件夹与源代码结构相对应：

```
tests/
├── agents/              # Agent 层测试
│   └── test_rag_agent.py
├── components/          # 组件层测试
│   ├── retrievers/      # 检索器测试
│   │   └── test_hybrid_retriever.py
│   ├── generators/      # 生成器测试
│   ├── classifier/      # 分类器测试
│   └── memory/          # 记忆组件测试
├── infra/              # 基础设施层测试
│   ├── db/             # 数据库测试
│   ├── llm/            # LLM 服务测试
│   └── utils/          # 工具类测试
├── api/                # API 层测试
│   └── test_routes.py
├── conftest.py         # 测试配置和共享 fixture
└── __init__.py
```

## 运行测试

### 运行所有测试

```bash
pytest
```

### 运行特定模块的测试

```bash
# 运行检索器测试
pytest tests/components/retrievers/ -v

# 运行 Agent 测试
pytest tests/agents/ -v

# 运行 API 测试
pytest tests/api/ -v
```

### 运行带标记的测试

```bash
# 只运行单元测试（默认）
pytest -m "not integration"

# 运行集成测试
pytest -m integration --run-integration

# 运行慢速测试
pytest -m slow
```

### 生成覆盖率报告

```bash
# 生成覆盖率报告
pytest --cov=src --cov-report=html

# 在终端显示覆盖率
pytest --cov=src --cov-report=term-missing
```

## 测试标记

- `@pytest.mark.unit`: 单元测试（默认）
- `@pytest.mark.integration`: 集成测试（需要外部服务）
- `@pytest.mark.slow`: 慢速测试
- `@pytest.mark.api`: API 测试

## 测试最佳实践

1. **单元测试**: 测试单个函数或类，不依赖外部服务
2. **集成测试**: 测试多个组件的协作，需要真实数据库或 API
3. **使用 fixture**: 共享测试数据和设置
4. **测试边界条件**: 测试极端情况和错误处理
5. **保持测试独立**: 每个测试应该独立运行

## 常用测试命令

```bash
# 显示详细输出
pytest -v

# 显示测试进度
pytest -v --tb=short

# 只显示失败的测试
pytest -v -x

# 失败时进入 pdb 调试
pytest -v --pdb

# 实时重新运行测试（需要 pytest-watch）
ptw
```
