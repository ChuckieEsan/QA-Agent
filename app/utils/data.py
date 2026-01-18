import hashlib
import re

def generate_doc_id(question: str, answer: str) -> str:
    """
    基于内容的 MD5 指纹生成
    将 问题+回答 拼接后计算 MD5，截取前 16 位作为 ID。
    """
    raw_str = str(question) + str(answer)
    return hashlib.md5(raw_str.encode('utf-8')).hexdigest()[:16]

