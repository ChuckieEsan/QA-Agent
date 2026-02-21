"""
测试配置组件
"""

import pytest
from src.config.setting import settings


class TestSettings:
    """测试配置"""

    def test_settings_loaded(self):
        """测试配置已加载"""
        assert settings is not None

    def test_project_name(self):
        """测试项目名称"""
        assert settings.project_name == "GovPulse"

    def test_paths_exist(self):
        """测试路径配置"""
        assert settings.paths.project_root.exists()

    def test_retriever_weights(self):
        """测试检索器权重配置"""
        # 验证权重总和接近 1.0（允许浮点误差）
        total_weight = (
            settings.retriever.weight_similarity +
            settings.retriever.weight_recency +
            settings.retriever.weight_length
        )
        assert abs(total_weight - 1.0) < 0.01
