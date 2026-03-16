"""
重排器模块

LangChain 兼容的重排器实现

包含：
- BGECompressor: LangChain 文档压缩器（非单例，每次请求实例化）
- create_bge_compressor: 创建压缩器的工厂函数

向后兼容：
- BGEReranker: 旧版单例类别名（已废弃，请使用 BGECompressor + BaseReranker）
"""

from .bge_reranker import BGERerankerCompressor, create_bge_compressor

__all__ = [
    "BGERerankerCompressor",
    "create_bge_compressor",
    # 向后兼容
    "BGEReranker",
]