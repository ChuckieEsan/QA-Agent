import hashlib
import re

def generate_doc_id(question: str, answer: str) -> str:
    """
    基于内容的 MD5 指纹生成
    将 问题+回答 拼接后计算 MD5，截取前 16 位作为 ID。
    """
    raw_str = str(question) + str(answer)
    return hashlib.md5(raw_str.encode('utf-8')).hexdigest()[:16]


def clean_text(text: str) -> str:
    """
    清理文本中的特殊字符（\n, \t, \xa0, 连续空格等）
    
    Args:
        text: 原始文本（可能为 None）
        
    Returns:
        清理后的纯文本
    """
    if not text:
        return ""
    
    # 1. 替换 HTML 特殊空格（\xa0 是不间断空格 &nbsp;）
    text = text.replace('\xa0', ' ').replace('\u200b', '')  # \u200b 是零宽空格
    
    # 2. 替换转义换行和制表符
    text = text.replace('\\n', ' ').replace('\\r', ' ').replace('\\t', ' ')
    
    # 3. 替换真实换行和制表符（如果 text_content 已经解码了转义）
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    
    # 4. 合并多个空格为一个
    text = re.sub(r'\s+', ' ', text)
    
    # 5. 去除首尾空格
    return text.strip()