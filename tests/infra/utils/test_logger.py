"""
测试工具类
"""

import pytest
from src.app.infra.utils.logger import get_logger


class TestLogger:
    """测试日志工具"""

    def test_logger_creation(self):
        """测试日志器创建"""
        logger = get_logger(__name__)
        assert logger is not None

    def test_logger_name(self):
        """测试日志器名称"""
        logger = get_logger("test.module")
        assert logger.name == "test.module"
