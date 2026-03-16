# 🚀 BaseLLMService 架构演进与优化蓝图 (Roadmap)

## 📌 定位与愿景
在当前的 `QA-Agent` 架构中，`BaseLLMService` 扮演着**防腐层（Anti-Corruption Layer, ACL）**与**大模型网关（LLM Gateway）**的角色。
未来的核心演进方向是：在不破坏 LangChain 表达式语言（LCEL）原生体验的前提下，将所有**横切关注点（Cross-cutting Concerns）**下沉到本服务中，实现对业务层完全透明的管控。

---

## 演进阶段一：全面可观测性与成本管控 (Observability & Billing)

**痛点**：目前直接代理 `_llm.invoke`，我们无法得知每个业务组件消耗了多少 Token，也无法针对用户/部门进行计费或限流。

**优化方案**：引入 LangChain 的 `Callbacks` 机制或拦截器模式，统一收集 Token 消耗。

```python
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from typing import Any

class TokenTrackingCallback(BaseCallbackHandler):
    """统一 Token 计费回调拦截器"""
    def __init__(self, user_id: str):
        self.user_id = user_id

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        # 从底层大模型的 response 中提取 token_usage
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            total_tokens = usage.get("total_tokens", 0)
            # TODO: 异步写入数据库/Redis进行扣费或统计
            print(f"💰 用户 {self.user_id} 本次消耗 Token: {total_tokens}")

# 在 BaseLLMService 中的改造：
class BaseLLMService(RunnableSerializable):
    def invoke(self, input: list[BaseMessage], config=None, user_id: str = "system"):
        # 动态注入拦截器，业务层无感知
        config = config or {}
        callbacks = config.get("callbacks",[])
        callbacks.append(TokenTrackingCallback(user_id=user_id))
        config["callbacks"] = callbacks
        
        return self._llm.invoke(input, config)
```

---

## 演进阶段二：高可用与智能路由 (High Availability & Fallback)

**痛点**：大模型 API（如 DeepSeek/OpenAI）偶尔会遇到 503 拥挤、限流（RateLimit）或超时。直接报错会导致整个 RAG 链路崩溃。

**优化方案**：利用 LangChain 原生的 `.with_fallbacks()` 结合自定义重试逻辑，打造坚不可摧的底层网关。

```python
class BaseLLMService(RunnableSerializable):
    def __init__(self, primary_provider="deepseek", fallback_provider="qwen", **kwargs):
        super().__init__(**kwargs)
        # 初始化主模型
        self._llm = self._create_model(primary_provider)
        # 初始化备用模型（如 Qwen 或 本地 Ollama 作为兜底）
        self._fallback_llm = self._create_model(fallback_provider)
        
        # 构建高可用链：主模型失败时，自动无缝切换备用模型
        self._ha_llm = self._llm.with_fallbacks(
            [self._fallback_llm],
            exceptions_to_handle=(TimeoutError, RateLimitError, APIConnectionError)
        )

    def invoke(self, input: list[BaseMessage], config=None):
        # 业务层调用的永远是具备自动兜底能力的组合模型
        return self._ha_llm.invoke(input, config)
```

---

## 演进阶段三：数据脱敏与安全合规 (Security & PII Masking)

**痛点**：政务问政数据中往往包含市民的真实姓名、身份证、电话等敏感信息（PII）。直接将这些数据发送给公有云大模型存在极高的合规风险。

**优化方案**：在输入大模型前进行正则/实体替换拦截，在输出后进行还原。

```python
class BaseLLMService(RunnableSerializable):
    
    def _mask_sensitive_data(self, messages: list[BaseMessage]) -> tuple[list[BaseMessage], dict]:
        """数据脱敏逻辑：提取并替换手机号/身份证等"""
        # 伪代码：将 "联系电话: 13800138000" 替换为 "联系电话: [PHONE_1]"
        # 返回脱敏后的消息列表，以及映射字典 {"[PHONE_1]": "13800138000"}
        pass

    def _unmask_sensitive_data(self, text: str, mapping: dict) -> str:
        """数据还原逻辑"""
        # 伪代码：将大模型生成的回复中的 "[PHONE_1]" 替换回真实号码
        pass

    def invoke(self, input: list[BaseMessage], config=None):
        # 1. 前置拦截：脱敏
        safe_input, mapping = self._mask_sensitive_data(input)
        
        # 2. 调用大模型
        response = self._llm.invoke(safe_input, config)
        
        # 3. 后置拦截：还原
        response.content = self._unmask_sensitive_data(response.content, mapping)
        return response
```

---

## 演进阶段四：解决 `with_structured_output` 的代理穿透问题 (Architecture Deep Dive)

**痛点**：目前 `BaseLLMService` 中的 `with_structured_output` 是直接 `return self._llm.with_structured_output(...)`。这会导致返回的 `Runnable` 丢失了我们在 `invoke` 中写的脱敏、计费等增强逻辑（**穿透效应**）。

**优化方案**：不直接返回底层对象，而是通过组合包装（Composition）返回一个包含拦截逻辑的自定义 `Runnable`。

```python
from langchain_core.runnables import RunnableLambda

class BaseLLMService(RunnableSerializable):

    def with_structured_output(self, schema: Type[BaseModel], method: str = "function_calling"):
        """
        拦截结构化输出方法，确保其依然享受 Service 级别的保护（如脱敏、高可用）
        """
        # 1. 获取底层的结构化输出链
        structured_llm = self._llm.with_structured_output(schema, method)
        
        # 2. 定义包装执行函数
        def _execute_with_interceptors(input_data: list[BaseMessage], config: dict):
            # 这里可以复用阶段一、三的拦截逻辑（如脱敏）
            safe_input, mapping = self._mask_sensitive_data(input_data)
            
            # 执行结构化输出
            result = structured_llm.invoke(safe_input, config)
            
            # (可选) 如果返回的是 Pydantic 对象且包含文本字段，可进行敏感词还原
            return result

        # 3. 返回被包装的 Runnable，完美兼容 LCEL 管道符 |
        return RunnableLambda(_execute_with_interceptors)
```

---

## 💡 面试官交流话术 (Interview Cheat Sheet)

在向面试官展示这套架构时，你可以这样引导对话：

> "在搭建 `QA-Agent` 的基础设施时，我特意设计了一层 `BaseLLMService` 作为**大模型网关（Gateway / ACL）**。
> 
> 虽然在项目初期（也就是目前），它看起来像是一个简单的代理封装，但我这么做是**为了给未来的企业级横向扩展留出接口**。在我的规划中，一旦业务规模扩大，我可以直接在这个 Service 层做几件事：
> 1. 利用 LangChain 的 **Callbacks** 无缝接入 Token 计费和日志埋点追踪。
> 2. 引入 **with_fallbacks** 实现开源本地模型（Ollama）与闭源商业模型（DeepSeek）的高可用主备切换。
> 3. 最重要的是，政务系统对数据隐私要求极高，我计划在这里实现**统一的 PII 脱敏拦截器**。
> 
> 为了解决 LangChain 代理模式下的‘穿透问题’（即调用 `with_structured_output` 会导致脱敏逻辑失效），我研究过底层源码，计划使用 `RunnableLambda` 对底层的链进行二次包装，确保所有的 LLM 请求绝对受控。这不仅保证了业务组件层的代码纯洁度，还实现了整个系统坚如磐石的安全合规底座。"