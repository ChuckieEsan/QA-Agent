"""
配置模块
提供应用配置管理
"""

from .setting import (
    Settings,
    PathConfig,
    ModelConfig,
    PostgresDBConfig,
    MilvusDBConfig,
    LLMConfig,
    RetrieverConfig,
)

# 全局配置实例
settings = Settings()

__all__ = [
    "Settings",
    "PathConfig",
    "ModelConfig",
    "PostgresDBConfig",
    "MilvusDBConfig",
    "LLMConfig",
    "RetrieverConfig",
    "settings",
]