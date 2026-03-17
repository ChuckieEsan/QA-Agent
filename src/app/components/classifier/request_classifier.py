"""
政务问政分类器
使用 langchain LCEL + PydanticOutputParser 实现

依赖注入模式：接受外部注入的 BaseLLMService 实例
"""

import traceback

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable
from src.app.infra.llm import BaseLLMService, create_llm_service
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum
from typing import Optional
from src.app.infra.llm import create_llm_service
from src.app.infra.utils.logger import get_logger

logger = get_logger(__name__)


class GovRequestType(Enum):
    """
    问政请求类型枚举

    泸州市网络问政平台的四种主要类型：
    1. 建议：对政府工作提出改进建议、意见
    2. 投诉：反映政府部门或工作人员的问题、不当行为
    3. 求助：请求政府帮助解决个人或家庭困难
    4. 咨询：询问政策、流程、办事指南等信息
    5. 其他：与问政内容无关
    """

    ADVICE = "advice"  # 建议
    COMPLAINT = "complaint"  # 投诉
    HELP = "help"  # 求助
    CONSULT = "consult"  # 咨询
    OTHER = "other"  # 其他
    
    @property
    def chinese(self):
        """根据成员名返回中文描述"""
        return {
            'ADVICE': '建议',
            'COMPLAINT': '投诉',
            'HELP': '求助',
            'CONSULT': '咨询',
            'OTHER': '其他',
        }[self.name]


class GovRequestClassifiedResult(BaseModel):
    request_type: GovRequestType = Field(description="问政请求类型")
    request_department: Optional[str] = Field(
        default="待确定",
        description="问政请求精确对应的市、区、县级主管部门或者相关单位。如果不确定具体的相关单位，请直接输出'待确定'",
    )
    request_city_department: Optional[str] = Field(
        default="待检索",
        description="问政请求对应的市级主管部门。如果不确定具体的管辖部门，请直接输出 '待检索'，不要瞎猜。",
    )

    class Config:
        # 允许字段别名（例如 'requestType' -> 'request_type'）
        populate_by_name = True


class GovRequestClassifier(BaseModel):
    """
    政务问政分类器

    使用 langchain LCEL 构建分类 Chain：
    - ChatPromptTemplate: 提示词模板
    - LLM.with_structured_output: 结构化输出

    依赖注入模式：通过 llm 字段接收外部注入的 BaseLLMService 实例
    """

    llm: BaseLLMService = Field(description="外部注入的大模型服务实例")

    # 配置字段
    top_k: int = Field(default=5, description="返回结果数量")

    # 允许传入任意类型的对象（避免 Pydantic 报错）
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context) -> None:
        """初始化后构建 LCEL 链"""
        self._build_chain()
        logger.info(
            f"GovRequestClassifier 初始化完成（使用模型: {getattr(self.llm, 'model_name', 'unknown')}）"
        )

    def _build_chain(self) -> None:
        """使用 langchain LCEL 构建分类 Chain"""

        # 1. 构建提示词模板
        self._prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
你是政务问政分类专家，请对以下市民诉求进行分类，并分析问政请求的紧急情况。
## 分类标准
1. 建议（advice）：对政府工作提出改进建议、意见
   - 关键词：建议、希望、改进、优化、提升
   - 示例："建议增加公交车班次"

2. 投诉（complaint）：反映政府部门或工作人员的问题、不当行为
   - 关键词：投诉、不满、违规、问题、差
   - 示例："投诉某部门办事效率低"

3. 求助（help）：请求政府帮助解决个人或家庭困难
   - 关键词：求助、帮忙、解决、困难、申请
   - 示例："我家房子漏水，请求帮助"

4. 咨询（consult）：询问政策、流程、办事指南等信息
   - 关键词：咨询、请问、如何、怎么、什么
   - 示例："咨询雨露计划申请条件"

5. 其他（other）：与政务无关的内容
   - 示例：1 + 1 等于几？今天天气怎么样

""",
                ),
                ("user", "{text}"),
            ]
        )

        # 2. 构建 LCEL Chain: prompt -> llm
        self._chain: RunnableSerializable = (
            self._prompt
            | self.llm.with_structured_output(
                GovRequestClassifiedResult, method="function_calling"
            )
        )

    def classify(self, text: str) -> GovRequestClassifiedResult:
        """
        分类问政请求

        Args:
            text: 市民诉求文本

        Returns:
            GovRequestClassifiedResult: 分类结果
        """
        try:
            result = self._chain.invoke(
                {"text": text}, config={"temperature": 0, "max_tokens": 500}
            )
            return result

        except Exception as e:
            logger.warning(f"分类失败: {e}，使用默认分类")
            logger.warning(traceback.format_exc())
            return GovRequestClassifiedResult(request_type=GovRequestType.CONSULT)


def create_gov_request_classifier(
    llm: Optional[BaseLLMService] = None,
    top_k: int = 5,
) -> GovRequestClassifier:
    """
    创建政务分类器的工厂函数

    Args:
        llm: 大模型实例（如果不提供则使用默认配置创建 BaseLLMService）
        top_k: 返回结果数量

    Returns:
        GovRequestClassifier 实例
    """
    if llm is None:
        llm = create_llm_service(provider_id="deepseek", model_name="deepseek-chat")

    return GovRequestClassifier(llm=llm, top_k=top_k)
