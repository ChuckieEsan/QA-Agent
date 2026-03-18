from typing import Annotated, Union
from langchain_core.tools import tool, InjectedToolCallId
from pydantic import BaseModel, Field

from src.app.components.retriever import (
    CasesVectorRetriever,
    PowersVectorRetriever,
    create_cases_retriever,
    create_powers_retriever,
)
from src.app.components.classifier import (
    GovRequestClassifier,
    GovRequestType,
    create_gov_request_classifier,
    GovRequestClassifiedResult,
)

from src.app.components.validator.answer_validator import (
    create_gov_answer_validator,
    GovAnswerValidatedResult,
    GovAnswerValidator,
)

from src.app.infra.utils.logger import get_logger
from langgraph.types import Command
from langchain.messages import AIMessage, ToolMessage
from langgraph.graph import END

logger = get_logger(__name__)

_powers_retriever: PowersVectorRetriever = create_powers_retriever(top_k=5)
_cases_retriever: CasesVectorRetriever = create_cases_retriever(top_k=5)
_gov_request_classifier: GovRequestClassifier = create_gov_request_classifier()
_gov_answer_validator: GovAnswerValidator = create_gov_answer_validator()


class RetrievePowersArgs(BaseModel):
    query: str = Field(
        ...,
        description="提炼后的用户诉求核心事项，需要包含事项主体和事项内容（如：'街道路灯不亮'、'工地拖欠农民工工资'），必须明确且精炼。",
    )
    top_k: int = Field(default=5, description="需要召回的权责清单数量，建议默认为 5。")


class RetrieveCasesArgs(BaseModel):
    query: str = Field(
        ...,
        description="用户咨询的具体政策、办理条件或纠纷关键词（如：'53岁女性补缴养老保险'、'灵活就业公积金提取'）。",
    )
    top_k: int = Field(
        default=5, description="需要召回的历史相似案例数量，建议默认为 5。"
    )


class ClassifiyGovRequestArgs(BaseModel):
    query: str = Field(..., description="用户原始的完整问政诉求文本")


class ValidateAnswerArgs(BaseModel):
    query: str = Field(..., description="用户原始的完整问政诉求文本。")
    context: str = Field(
        ...,
        description="你基于检索工具获取到的核心政策依据或历史案例（请提炼核心事实传入，不要为空）。",
    )
    draft_answer: str = Field(..., description="你准备回复给用户的【草稿回答】。")


@tool("retrieve_powers_tool", args_schema=RetrievePowersArgs)
def retrieve_powers_tool(query: str, top_k: int = 5) -> str:
    """
    【权责清单与部门流转检索工具】
    核心用途：当你需要准确判断某个用户诉求（投诉、举报、求助）应该由哪个具体的政府部门（如：人社局、城管局、住建局）管辖时，必须调用此工具。
    返回值：包含相似度、相关的权责名称、行使主体（所属部门）以及具体的管理权限说明。
    请根据返回结果中的“负责部门”，指导用户或决定派单去向。
    """
    logger.info(f"正在检索权责清单，提取管辖部门 | Query: {query}")

    docs = _powers_retriever.invoke(query)[:top_k]

    if not docs:
        return f"未找到与 '{query}' 相关的权责清单。请尝试提取更核心的业务关键词重新检索，或者建议派发综合工单。"

    results = []
    for i, doc in enumerate(docs):
        results.append(
            f"【相似度】{doc.metadata['similarity']}，【匹配权责 {i+1}】{doc.page_content}"
        )

    return "\n\n".join(results)


@tool("retrieve_cases_tool", args_schema=RetrieveCasesArgs)
def retrieve_cases_tool(query: str, top_k: int = 5) -> str:
    """
    【历史问政案例与政策检索工具】
    核心用途：当用户咨询具体的政务办理条件、政策细节、所需材料，或者你需要解答相似的业务问题时，必须调用此工具。
    返回值：过往相似的市民问政历史记录。包含相似度、历史问题及官方答复。
    """
    logger.info(f"正在检索历史办理案例，提取政策依据 | Query: {query}")

    docs = _cases_retriever.invoke(query)[:top_k]

    if not docs:
        return f"未找到与 '{query}' 相关的历史办理案例。请尝试换一个相近的搜索词，或者告知用户暂无相关政策数据，询问是否需要转人工处理。"

    results = []
    for i, doc in enumerate(docs):
        # 利用我们在底层检索器里存入的 QA Metadata
        results.append(
            f"【相似度】{doc.metadata['similarity']}，【相似案例 {i+1}】{doc.page_content}"
        )

    return "\n\n".join(results)


@tool("classify_gov_request_tool", args_schema=ClassifiyGovRequestArgs)
def classify_gov_request_tool(query: str) -> str:
    """
    【用户诉求意图分析工具】
    核心用途：在处理用户全新的问政诉求时，这是你应该调用的**第一个工具**。
    功能：它可以精准分析用户是在“咨询”、“投诉”、“建议”、“求助”还是“其他”，并初步评估可能涉及的管辖部门
    返回值：包含确定的请求类型（request_type）以及严格的下一步行动建议。
    注意：请严格根据本工具返回的“系统建议”来决定后续调用哪个检索工具。
    """
    logger.info(f"正在对用户诉求进行意图分类 | Query: {query}")

    result: GovRequestClassifiedResult = _gov_request_classifier.classify(query)

    req_type = result.request_type
    req_dept = result.request_department
    req_city_dept = result.request_city_department

    action_advice = ""

    if req_type == GovRequestType.CONSULT:
        action_advice = "当前意图为【咨询】。请立刻调用 `retrieve_cases_tool` 检索政策和历史类似问题的解答，为用户提供权威参考。"
    elif req_type in [
        GovRequestType.COMPLAINT,
        GovRequestType.ADVICE,
        GovRequestType.HELP,
    ]:
        action_advice = f"当前意图为【{req_type.chinese}】。因为涉及权责划分，请你务必先调用 `retrieve_powers_tool` 检索权责清单以确定准确的管辖部门。确认部门后，再考虑是否调用 `create_work_order` 派发工单。"
    elif req_type == GovRequestType.OTHER:
        action_advice = "当前意图为【其他/闲聊】。无需调用任何检索工具，请直接以政务助手的身份礼貌回复用户。"
    else:
        action_advice = "请综合利用检索工具寻找答案。"

    # 3. 组装返回给大模型的结构化纯文本
    output = (
        f"【意图分类结果】\n"
        f"- 问政类型: {req_type.chinese} ({req_type})\n"
        f"- 初步评估相关单位: {req_dept}\n"
        f"- 初步评估市级部门: {req_city_dept}\n"
        f"\n【系统建议 (非常重要)】:\n{action_advice}"
    )

    return output


@tool("validate_answer_tool", args_schema=ValidateAnswerArgs)
async def validate_answer_tool(
    query: str,
    context: str,
    draft_answer: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Union[str, Command]:
    """
    【回复质量与安全审查工具 (最终回复前必须调用)】
    核心用途：当你收集完信息，并**准备向用户输出最终答案之前**，你必须调用此工具对你的“草稿回答”进行审核。
    功能：评估你的回答是否包含敏感词、是否出现了脱离上下文的捏造政策（幻觉）。
    注意：
    1. 如果工具返回【通过】，你可以直接将你的草稿作为最终回复输出。
    2. 如果工具返回【不通过】，你**绝对不能**将草稿发给用户！必须仔细阅读“修改建议”，修正你的草稿，并可以再次调用此工具直到通过为止。
    """
    logger.info(f"正在校验大模型草稿...")

    # 调用同步验证逻辑
    result: GovAnswerValidatedResult = await _gov_answer_validator.validate(
        answer=draft_answer, query=query, context=context
    )

    # 组装返回给大模型的反馈意见
    if result.is_passed:
        logger.info("草稿校验通过！")
        # 这里直接返回结果，不再重复输出消耗 token
        return Command(
            update={
                "messages": [
                    # 闭合刚才的 tool_call
                    ToolMessage(
                        content="【系统底层指令】校验已通过，强制阻断 LLM 二次生成。",
                        tool_call_id=tool_call_id,
                    ),
                    # 然后再把拦截下来的最终回答塞进去
                    AIMessage(content=draft_answer),
                ]
            },
        )
    else:
        logger.warning(f"草稿校验被拦截，要求大模型重写。原因: {result.suggestion}")
        return (
            "【审核结果】: 不通过 (存在严重幻觉或违规)\n"
            f"【拦截原因与修改建议】: {result.suggestion}\n"
            "【系统指示】: 你的草稿已被拦截！请严格根据上述修改建议，重新思考并修改你的回答。修改后可再次调用本工具检查，或者如无法解答，请调用 create_work_order 派发工单。"
        )
