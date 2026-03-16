"""
BGE 重排模型使用示例

新架构：
- Infra 层：BaseRerankerClient（单例，线程安全）
- Components 层：BGERerankerCompressor（非单例，每次请求实例化）
"""
import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())

from langchain_core.documents import Document

from src.app.infra.reranker import BaseRerankerClient, get_reranker_client
from src.app.components.reranker import BGERerankerCompressor, create_bge_compressor
from src.config.setting import settings
from src.app.infra.utils.logger import get_logger

logger = get_logger(__name__)


def demo_bge_reranker():
    """
    演示 BGE 重排模型的使用（新架构）
    """
    logger.info("="*60)
    logger.info("BGE 重排模型演示")
    logger.info("="*60)

    # 1. 获取单例 client（线程安全）
    logger.info("获取 BGE Reranker 客户端（单例）...")
    reranker_client = get_reranker_client()

    # 验证单例
    reranker_client_2 = get_reranker_client()
    logger.info(f"单例验证: {reranker_client is reranker_client_2}")  # 应为 True

    # 2. 创建 compressor（非单例，每次请求实例化）
    logger.info("创建 BGE 压缩器（非单例）...")
    compressor = create_bge_compressor(
        bge_client=reranker_client,
        top_k=3,
        min_score=0.0,
    )

    # 验证 compressor 非单例
    compressor_2 = create_bge_compressor(bge_client=reranker_client)
    logger.info(f"Compressor 非单例验证: {compressor is compressor_2}")  # 应为 False

    # 示例查询和文档
    query = "如何申请政府信息公开"
    documents = [
        Document(
            page_content="政府信息公开条例规定，公民、法人或其他组织可以向行政机关申请获取相关政府信息。申请人需填写申请表，明确所需信息的内容和用途。",
            metadata={"title": "政府信息公开申请流程"}
        ),
        Document(
            page_content="政府信息公开是提高政府工作透明度的重要措施。政府部门应主动公开政策法规、行政决策过程等信息。",
            metadata={"title": "政府信息公开制度概述"}
        ),
        Document(
            page_content="个人信息保护法规定了个人对其信息的控制权。任何组织或个人不得非法收集、使用他人个人信息。",
            metadata={"title": "个人信息保护法要点"}
        ),
        Document(
            page_content="行政诉讼法适用于公民对行政机关的行为提起诉讼的情况。起诉人需在法定期限内提交起诉状及相关证据材料。",
            metadata={"title": "行政诉讼法基本规定"}
        ),
    ]

    logger.info(f"查询: {query}")
    logger.info(f"待重排文档数量: {len(documents)}")

    # 3. 执行重排（使用 compressor）
    logger.info("执行重排...")
    reranked_results = compressor.compress_documents(documents, query)

    logger.info("重排结果:")
    for i, result in enumerate(reranked_results, 1):
        logger.info(f"  {i}. [{result.metadata.get('title', 'N/A')}] 重排得分: {result.metadata.get('rerank_score', 0):.4f}")
        logger.info(f"     内容: {result.page_content[:100]}...")

    # 4. 单独计算得分示例（使用 client）
    logger.info("单个文档得分计算示例:")
    sample_text = "政府信息公开申请需要填写详细的申请表格"
    scores = reranker_client.compute_score(query, [sample_text])
    logger.info(f"   查询: {query}")
    logger.info(f"   文本: {sample_text}")
    logger.info(f"   相关性得分: {scores[0]:.4f}")

    logger.info("BGE 重排模型演示完成")


if __name__ == "__main__":
    demo_bge_reranker()