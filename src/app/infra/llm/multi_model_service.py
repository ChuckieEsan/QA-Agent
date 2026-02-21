"""
多模型 LLM 服务管理器
采用配置驱动的模型注册表模式，支持灵活的多模型配置

架构设计：
1. 模型配置（ModelConfig）：定义每个模型的独立配置（api_key、参数等）
2. LLM 服务（LLMService）：封装单个模型的调用逻辑
3. 模型注册表（ModelRegistry）：单例模式，集中管理所有模型实例
4. 便捷访问：通过预定义函数或用途枚举获取服务

使用方式：
    # 方式1：直接获取（推荐）
    llm = get_heavy_llm_service()

    # 方式2：通过用途获取（更语义化）
    llm = get_llm_service_by_purpose(ModelPurpose.GENERATION)

    # 获取模型配置
    config = llm.get_config()
    temperature = config.temperature
"""

import dashscope
from typing import Dict, Optional
from enum import Enum
from src.config.setting import settings
from src.app.infra.utils.logger import get_logger

logger = get_logger(__name__)


class ModelPurpose(Enum):
    """模型用途枚举"""
    GENERATION = "generation"      # 主模型：生成复杂回答
    CLASSIFICATION = "classification"  # 轻量模型：分类、校验
    OPTIMIZATION = "optimization"  # 优化模型：Prompt优化、Agent决策


class ModelConfig:
    """单个模型的配置"""

    def __init__(
        self,
        name: str,
        api_key: str,
        purpose: ModelPurpose,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        top_p: float = 0.9,
    ):
        self.name = name
        self.api_key = api_key
        self.purpose = purpose
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p


class LLMService:
    """单个 LLM 服务实例"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model_name = config.name

        # 配置 API 密钥
        dashscope.api_key = config.api_key

        logger.info(
            f"🔄 初始化 {config.purpose.value} LLM 服务: {config.name}"
        )

    def get_model_name(self) -> str:
        """获取模型名称"""
        return self.model_name

    def get_config(self) -> ModelConfig:
        """获取模型配置"""
        return self.config


class ModelRegistry:
    """模型注册表 - 单例模式"""

    _instance: Optional["ModelRegistry"] = None
    _services: Dict[str, LLMService] = {}

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        # 从配置中注册所有模型
        self._register_models_from_config()
        self._initialized = True
        logger.info("✅ 模型注册表初始化完成")

    def _register_models_from_config(self):
        """从 settings 配置中注册模型"""

        # 注册主模型（生成）
        heavy_config = ModelConfig(
            name=settings.llm.heavy_model_name,
            api_key=settings.llm.api_key,  # 可以扩展为 settings.llm.heavy_api_key
            purpose=ModelPurpose.GENERATION,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
            top_p=settings.llm.top_p,
        )
        self._register_service("heavy", LLMService(heavy_config))

        # 注册轻量模型（分类）
        light_config = ModelConfig(
            name=settings.llm.light_model_name,
            api_key=settings.llm.api_key,  # 可以扩展为 settings.llm.light_api_key
            purpose=ModelPurpose.CLASSIFICATION,
            temperature=0.1,  # 分类任务使用更低的 temperature
            max_tokens=500,   # 分类任务 token 限制更低
            top_p=0.9,
        )
        self._register_service("light", LLMService(light_config))

        # 注册优化模型（可选，默认与轻量模型共用）
        if hasattr(settings.llm, "optimizer_model_name"):
            optimizer_config = ModelConfig(
                name=settings.llm.optimizer_model_name,
                api_key=settings.llm.api_key,
                purpose=ModelPurpose.OPTIMIZATION,
                temperature=0.3,
                max_tokens=1000,
                top_p=0.9,
            )
            self._register_service("optimizer", LLMService(optimizer_config))
        else:
            # 与轻量模型共用
            self._services["optimizer"] = self._services["light"]
            logger.info("  → 优化模型与轻量模型共用")

    def _register_service(self, key: str, service: LLMService):
        """注册服务实例"""
        self._services[key] = service
        logger.info(
            f"  → 已注册: {key} ({service.get_model_name()}, "
            f"用途: {service.get_config().purpose.value})"
        )

    def get_service(self, key: str) -> LLMService:
        """通过键名获取服务"""
        if key not in self._services:
            raise ValueError(f"未找到模型服务: {key}")
        return self._services[key]

    def get_by_purpose(self, purpose: ModelPurpose) -> LLMService:
        """通过用途获取服务"""
        # 映射用途到服务键
        purpose_map = {
            ModelPurpose.GENERATION: "heavy",
            ModelPurpose.CLASSIFICATION: "light",
            ModelPurpose.OPTIMIZATION: "optimizer",
        }
        key = purpose_map.get(purpose)
        if not key or key not in self._services:
            raise ValueError(f"未找到用途为 {purpose.value} 的模型服务")
        return self._services[key]


# ==================== 全局注册表实例 ====================

_registry = ModelRegistry()


# ==================== 工具函数 ====================

def get_heavy_llm_service() -> LLMService:
    """
    获取主 LLM 服务（生成复杂回答）

    使用场景：
    - LLMGenerator：生成最终回答
    - 复杂的文本生成任务
    """
    return _registry.get_service("heavy")


def get_light_llm_service() -> LLMService:
    """
    获取轻量 LLM 服务（分类/校验等简单任务）

    使用场景：
    - GovClassifier：问政分类
    - 回答质量校验
    - 简单的文本分析任务
    """
    return _registry.get_service("light")


def get_optimizer_llm_service() -> LLMService:
    """
    获取优化 LLM 服务（Agent 决策、Prompt 优化）

    使用场景：
    - Agent 意图分析
    - Prompt 重写
    - 中等复杂度的分析任务
    """
    return _registry.get_service("optimizer")


def get_llm_service(purpose: ModelPurpose) -> LLMService:
    """
    根据用途获取 LLM 服务

    Args:
        purpose: 模型用途枚举

    Returns:
        LLMService: 对应的 LLM 服务实例
    """
    return _registry.get_by_purpose(purpose)
