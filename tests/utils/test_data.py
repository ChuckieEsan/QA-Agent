import pytest
from app.utils.data import generate_doc_id, clean_text


def test_generate_doc_id_format():
    """测试 ID 的格式规范 (类型和长度)"""
    doc_id = generate_doc_id("测试问题", "测试回答")
    assert isinstance(doc_id, str)
    assert len(doc_id) == 16

def test_generate_doc_id_consistency():
    """测试幂等性：相同输入必须生成相同 ID"""
    q = "公积金如何提取"
    a = "咨询相关部门"
    
    id1 = generate_doc_id(q, a)
    id2 = generate_doc_id(q, a)
    
    assert id1 == id2

@pytest.mark.parametrize("q1, a1, q2, a2", [
    ("公积金", "回答", "社保", "回答"),      # 问题不同
    ("公积金", "回答A", "公积金", "回答B"),  # 回答不同
    ("公积金", "回答", " 公积金", "回答"),   # 有无空格 (MD5 对空格敏感)
    ("", "", "a", ""),                      # 边界：空字符串
])
def test_generate_doc_id_uniqueness(q1, a1, q2, a2):
    """测试唯一性：任何微小差异都应生成不同 ID"""
    id1 = generate_doc_id(q1, a1)
    id2 = generate_doc_id(q2, a2)
    assert id1 != id2


@pytest.mark.parametrize("input_text, expected, description", [
    # === 1. 基础情况 ===
    ("Hello World", "Hello World", "普通文本不应改变"),
    ("", "", "空字符串应返回空"),
    (None, "", "None 应返回空字符串"),
    
    # === 2. 空格合并与去除首尾 ===
    ("  Hello   World  ", "Hello World", "应去除首尾空格并合并中间多余空格"),
    ("   ", "", "纯空格字符串应变为空"),

    # === 3. 真实换行符与制表符 ===
    ("Line1\nLine2", "Line1 Line2", "真实换行符应转为空格"),
    ("Col1\tCol2", "Col1 Col2", "真实制表符应转为空格"),
    ("Line1\r\nLine2", "Line1 Line2", "Windows换行符应转为空格"),

    # === 4. 转义字符 (字面量的 \n, \t) ===
    # 注意：在 Python 字符串中表示字面量 \n 需要写成 \\n
    ("Line1\\nLine2", "Line1 Line2", "转义换行符应转为空格"),
    ("Col1\\tCol2", "Col1 Col2", "转义制表符应转为空格"),

    # === 5. 特殊 Unicode 字符 ===
    ("Non\xa0Breaking", "Non Breaking", "不间断空格(\xa0)应转为普通空格"),
    ("Zero\u200bWidth", "ZeroWidth", "零宽空格(\u200b)应被删除"),

    # === 6. 综合混合场景 (The Kitchen Sink) ===
    (
        " \n  Start \xa0 Middle\\nContent \t\u200b End  \n", 
        "Start Middle Content End", 
        "混合复杂场景测试"
    ),
])
def test_clean_text(input_text, expected, description):
    """
    测试 clean_text 函数的各种清洗逻辑
    参数 description 仅用于代码可读性，pytest 运行时不会显示
    """
    assert clean_text(input_text) == expected
