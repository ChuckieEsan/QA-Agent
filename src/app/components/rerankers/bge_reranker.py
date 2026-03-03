"""
BGE 重排模型组件

实现了基于 BGE-Reranker 的文本重排功能
"""
import torch
from typing import List, Dict, Any
from pathlib import Path

from FlagEmbedding import FlagReranker
from src.config.setting import settings
from .base_reranker import BaseReranker


class BGEReranker(BaseReranker):
    """
    BGE 重排模型包装类

    该类封装了 BGE-Reranker 模型的功能，提供简单的重排接口
    """

    def __init__(self, model_path: Path = None):
        """
        初始化 BGE 重排模型

        Args:
            model_path: 重排模型路径，如果为 None 则使用 settings 中的配置
        """
        if model_path is None:
            model_path = settings.models.reranker_model_path

        print(f"🔄 [BGEReranker] 加载重排模型: {model_path} ...")

        # 检查模型路径是否存在
        if not model_path.exists():
            raise FileNotFoundError(f"重排模型路径不存在: {model_path}")

        # 加载重排模型
        self.reranker = FlagReranker(
            model_name_or_path=str(model_path),
            use_fp16=torch.cuda.is_available()  # 如果有 GPU 则使用 fp16 加速
        )

        self.model_path = model_path
        print(f"✅ [BGEReranker] 重排模型加载完成")

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        对文档进行重排

        Args:
            query: 查询文本
            documents: 待重排的文档列表，每个文档应包含 'content' 键
            top_k: 返回前 K 个结果，如果为 None 则返回全部结果

        Returns:
            重排后的文档列表，按相关性降序排列，每个文档增加了 'rerank_score' 字段
        """
        if not documents:
            return []

        # 准备重排数据
        texts = []
        for doc in documents:
            # 尝试从不同字段获取文本内容
            content = doc.get('content') or doc.get('text') or doc.get('entity', {}).get('text', '')
            if content:
                texts.append(content)
            else:
                texts.append(str(doc))  # 如果没有找到合适的文本字段，则转换整个文档为字符串

        # 创建查询-文档对
        pairs = [[query, text] for text in texts]

        # 执行重排评分
        scores = self.reranker.compute_score(pairs)

        # 将得分与原文档关联
        reranked_docs = []
        for i, doc in enumerate(documents):
            updated_doc = doc.copy()  # 复制原文档
            updated_doc['rerank_score'] = float(scores[i])  # 添加重排得分
            reranked_docs.append(updated_doc)

        # 按重排得分降序排序
        reranked_docs.sort(key=lambda x: x['rerank_score'], reverse=True)

        # 返回前 top_k 个结果
        if top_k is not None:
            return reranked_docs[:top_k]

        return reranked_docs

    def compute_score(self, query: str, text: str) -> float:
        """
        计算单个查询与文本的相关性得分

        Args:
            query: 查询文本
            text: 待评分文本

        Returns:
            相关性得分
        """
        score = self.reranker.compute_score([[query, text]])
        return float(score[0])