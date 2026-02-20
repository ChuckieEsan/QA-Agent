"""
GovPulse 测试配置
包含所有测试的共享配置
"""

import pytest
from pathlib import Path
from src.config.setting import settings


# ==================== Fixtures ====================

@pytest.fixture(scope="session")
def test_settings():
    """测试用配置"""
    return settings


@pytest.fixture(scope="function")
def sample_query():
    """示例查询"""
    return "雨露计划什么时候发放？"


@pytest.fixture(scope="function")
def sample_history():
    """示例对话历史"""
    return [
        {"role": "user", "content": "什么是雨露计划？"},
        {"role": "assistant", "content": "雨露计划是针对农村贫困家庭的扶持政策..."}
    ]


# ==================== 标记配置 ====================

def pytest_configure(config):
    """配置 pytest 标记"""
    config.addinivalue_line(
        "markers", "integration: 标记为集成测试"
    )
    config.addinivalue_line(
        "markers", "slow: 标记为慢速测试"
    )
    config.addinivalue_line(
        "markers", "api: 标记为 API 测试"
    )


# ==================== 常量 ====================

TEST_QUERIES = [
    "雨露计划什么时候发放？",
    "公积金提取需要什么条件？",
    "创业补贴怎么申请？",
    "医保报销流程是什么？",
    "小微企业税收优惠政策？",
]

MIN_SIMILARITY_THRESHOLD = 0.5
MAX_RETRIEVAL_TIME = 5.0  # 秒
