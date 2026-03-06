"""
分类器组件模块
"""

from src.app.components.classifier.base_classifier import BaseClassifier, GovRequestType
from src.app.components.classifier.request_classifier import GovRequestClassifier

__all__ = [
    "BaseClassifier",
    "GovRequestClassifier",
    "GovRequestType",
]
