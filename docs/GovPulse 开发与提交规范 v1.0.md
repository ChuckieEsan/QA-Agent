# GovPulse 开发与提交规范 v1.0

## 1. 分支管理规范 (Branching Strategy)

为了保证代码库的整洁和稳定，我们采用简化版的 **GitHub Flow**。

### 1.1 分支命名规则
分支名应全小写，使用连字符 `-` 分隔。格式：`type/description`

| 类型       | 前缀        | 说明                   | 示例                         |
| :--------- | :---------- | :--------------------- | :--------------------------- |
| **新功能** | `feat/`     | 开发新的功能模块       | `feat/milvus-connection`     |
| **修复**   | `fix/`      | 修复 Bug               | `fix/data-ingest-crash`      |
| **文档**   | `docs/`     | 仅修改文档             | `docs/update-readme`         |
| **重构**   | `refactor/` | 代码重构（无功能变动） | `refactor/project-structure` |
| **优化**   | `perf/`     | 性能优化               | `perf/rerank-speedup`        |

### 1.2 开发流程
1.  **切分支**：永远不要直接在 `main` 分支修改代码。
    ```bash
    git checkout -b feat/add-redis-cache
    ```
2.  **开发**：编写代码，进行本地测试。
3.  **提交**：遵守下文的 Commit Message 规范。
4.  **合并**：推送到远程并发起 Pull Request (PR)。

---

## 2. Commit Message 提交规范 (核心)

我们采用业界最通用的 **Conventional Commits (约定式提交)** 规范。

### 2.1 消息格式
```text
<Type>(<Scope>): <Subject>

<Body> (可选)

<Footer> (可选)
```

### 2.2 Header 说明
Header 是必须的，且不超过 50 个字符。

*   **Type (类型)**：
    *   `feat`: ✨ 新功能 (Feature)
    *   `fix`: 🐛 修复 Bug
    *   `docs`: 📚 文档变更
    *   `style`: 💎 代码格式（不影响代码运行的变动，如空格、格式化）
    *   `refactor`: ♻️ 代码重构（既不是新增功能，也不是修改bug）
    *   `perf`: 🚀 性能优化
    *   `test`: 🧪 增加测试或修改测试
    *   `chore`: 🔧 构建过程或辅助工具的变动 (如 config.py, .gitignore)
    *   `ci`: 👷 CI/CD 配置文件修改

*   **Scope (范围)**：
    *   用于说明改动的影响范围（括号内），例如：`ingest`, `api`, `milvus`, `rag`, `config`。
    *   如果改动太杂，可以省略或写 `*`。

*   **Subject (主题)**：
    *   简短描述。
    *   **使用祈使句** (例如 "Add feature" 而不是 "Added feature")。
    *   **结尾不要加句号**。

### 2.3 Body 说明 (可选)
*   详细描述**为什么**修改，以及**怎么**修改的。
*   每行大约 72 个字符换行。

---

## 3. 实战示例 (GovPulse 项目)

### ✅ 示例 1：添加新功能
```text
feat(rag): add hybrid search logic with bm25

Implemented a hybrid retrieval strategy combining vector search (BGE-M3) 
and keyword search (BM25) to improve recall on specific terms.
```

### ✅ 示例 2：修复 Bug
```text
fix(ingest): handle missing 'department' column in excel

Previously, the ingestion script crashed if the source Excel file 
did not contain a 'department' header. Added a default fallback value.
```

### ✅ 示例 3：配置文件调整
```text
chore(config): refactor path handling using pathlib

Moved hardcoded paths from ingest.py to app/core/config.py 
to support cross-platform compatibility (Windows/Linux).
```

### ✅ 示例 4：文档更新
```text
docs: update readme with quick start guide
```

---

## 4. 提交前的检查清单 (Checklist)

在执行 `git commit` 之前，请自问：

1.  **原子性**：这次提交是不是只做了一件事？（不要把修复 Bug 和重构代码混在一个 Commit 里）。
2.  **代码风格**：是否已经运行了 Format 工具？
    *   建议安装 `pre-commit` 钩子（下文介绍）。
3.  **敏感信息**：有没有误提交 API Key 或密码？（请检查 `config.py` 或 `.env` 是否被忽略）。

---

## 5. 进阶：自动化工具配置 (Python 推荐)

为了强制执行这些规范，作为一名 Python 工程师，建议在项目中配置 **pre-commit**。

### 5.1 安装 pre-commit
```bash
uv add pre-commit  # 或者 pip install pre-commit
```

### 5.2 创建 `.pre-commit-config.yaml`
在项目根目录创建此文件，内容如下：

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace   # 自动去除行尾空格
      - id: end-of-file-fixer     # 自动确保文件以空行结尾
      - id: check-yaml            # 检查 yaml 语法
      - id: check-added-large-files # 防止提交大文件 (>500KB)

  - repo: https://github.com/psf/black
    rev: 24.2.0
    hooks:
      - id: black  # 自动格式化 Python 代码

  # 可选：强制检查 commit message 格式
  - repo: https://github.com/compilerla/conventional-pre-commit
    rev: v3.1.0
    hooks:
      - id: conventional-pre-commit
        stages: [commit-msg]
```

### 5.3 启用钩子
```bash
pre-commit install
pre-commit install --hook-type commit-msg
```

**效果**：
以后你每次运行 `git commit` 时：
1.  **Black** 会自动帮你格式化代码（如果没有格式化好，提交会失败，帮你改好后你需要再 add 一次）。
2.  **Hooks** 会检查你是否提交了超大文件（比如模型权重 `model.safetensors`，这种文件**绝对不能**提交到 Git，应该用 `.gitignore` 忽略）。
3.  **Commit Msg** 会检查你的写的是不是符合 `feat: ...` 格式。

---

## 6. `.gitignore` 建议 (针对本项目)

确保你的 Git 仓库里没有垃圾文件。

```gitignore
# Python
__pycache__/
*.py[cod]
.venv/
.env

# 数据 & 模型 (非常重要，不要提交大文件！)
data/raw/*.xlsx
data/milvus_db/
models/
*.bak

# IDE
.vscode/
.idea/

# Logs
*.log
```

---

### 总结

作为算法应用工程师，你的 Git 历史就是你的**思维快照**。

*   **如果你写**：`commit -m "fix"` -> **不仅难看，而且由于没说什么 bug，后面出了问题没法回滚。**
*   **如果你写**：`fix(ingest): resolve OOM issue when batch size > 32` -> **专业，即使代码逻辑有问题，别人也知道你的意图。**