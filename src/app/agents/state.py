"""
State 定义 - 政务问答 Agent 的共享状态

使用 TypedDict 定义全局共享状态，所有 Node 通过修改 State 来实现数据流转
"""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
from dataclasses import dataclass, field


class AppealState(TypedDict, total=False):
    """
    工单状态 - 对应 README 中的标准化工单

    使用 TypedDict 定义，支持:
    - raw_query: 原始诉求
    - cleaned_query: 清洗后的诉求
    - desensitized_query: 脱敏后的诉求
    - extracted_elements: 提取的要素
    - appeal_type: 诉求类型
    - urgency_level: 紧急程度
    - department: 办理部门
    - is_invalid: 是否无效诉求
    - retrieval_results: 检索结果
    - generated_answer: 生成的回答
    - validation_result: 验证结果
    - final_response: 最终回复
    - error_message: 错误信息
    - current_step: 当前步骤
    """
    # 输入
    raw_query: str

    # 预处理阶段
    cleaned_query: str
    desensitized_query: str
    extracted_elements: Dict[str, Any]

    # 分类阶段
    appeal_type: str
    urgency_level: str
    department: str
    is_invalid: bool

    # 检索阶段
    retrieval_results: List[Dict[str, Any]]

    # 生成阶段
    generated_answer: str

    # 验证阶段
    validation_result: Dict[str, Any]

    # 输出
    final_response: str

    # 元数据
    error_message: str
    current_step: str


# 简化的消息类型定义
Messages = List[Dict[str, Any]]


@dataclass
class ProcessingResult:
    """处理结果数据类"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
