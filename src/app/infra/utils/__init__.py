"""
工具函数模块
提供日志、设备检测等通用工具
"""

from .logger import get_logger, setup_logging, PerformanceLogger
from .system_utils import get_device
from .data_utils import generate_doc_id, clean_text

__all__ = [
    # Logger
    "get_logger",
    "setup_logging",
    "PerformanceLogger",
    # System
    "get_device",
    # Data
    "generate_doc_id",
    "clean_text",
]