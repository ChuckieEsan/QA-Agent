"""
BGE 重排模型使用示例
"""
import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())

from src.app.components.rerankers import BGEReranker
from src.config.setting import settings
from src.app.infra.utils.logger import get_logger

logger = get_logger(__name__)


def demo_bge_reranker():
    """
    演示 BGE 重排模型的使用
    """
    logger.info("="*60)
    logger.info("BGE 重排模型演示")
    logger.info("="*60)

    # 初始化 BGE 重排模型
    logger.info("初始化 BGE 重排模型...")
    reranker = BGEReranker()

    # 示例查询和文档
    query = "如何申请政府信息公开"
    documents = [
        {"content": "政府信息公开条例规定，公民、法人或其他组织可以向行政机关申请获取相关政府信息。申请人需填写申请表，明确所需信息的内容和用途。", "title": "政府信息公开申请流程"},
        {"content": "政府信息公开是提高政府工作透明度的重要措施。政府部门应主动公开政策法规、行政决策过程等信息。", "title": "政府信息公开制度概述"},
        {"content": "个人信息保护法规定了个人对其信息的控制权。任何组织或个人不得非法收集、使用他人个人信息。", "title": "个人信息保护法要点"},
        {"content": "行政诉讼法适用于公民对行政机关的行为提起诉讼的情况。起诉人需在法定期限内提交起诉状及相关证据材料。", "title": "行政诉讼法基本规定"}
    ]

    logger.info(f"查询: {query}")
    logger.info(f"待重排文档数量: {len(documents)}")

    # 执行重排
    logger.info("执行重排...")
    reranked_results = reranker.rerank(query, documents)

    logger.info("重排结果:")
    for i, result in enumerate(reranked_results, 1):
        logger.info(f"  {i}. [{result['title']}] 重排得分: {result['rerank_score']:.4f}")
        logger.info(f"     内容: {result['content'][:100]}...")

    # 单独计算得分示例
    logger.info("单个文档得分计算示例:")
    sample_text = "政府信息公开申请需要填写详细的申请表格"
    score = reranker.compute_score(query, sample_text)
    logger.info(f"   查询: {query}")
    logger.info(f"   文本: {sample_text}")
    logger.info(f"   相关性得分: {score:.4f}")

    logger.info("BGE 重排模型演示完成")


if __name__ == "__main__":
    demo_bge_reranker()