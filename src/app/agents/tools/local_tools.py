from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.app.components.retriever import (
    CasesVectorRetriever,
    PowersVectorRetriever,
    create_cases_retriever,
    create_powers_retriever,
)
from src.app.components.classifier import GovRequestClassifier, GovRequestType ,create_gov_request_classifier, GovRequestClassifiedResult


_powers_retriever: PowersVectorRetriever = create_powers_retriever(top_k=5)
_cases_retriever: CasesVectorRetriever = create_cases_retriever(top_k=5)
_gov_request_classifier: GovRequestClassifier = create_gov_request_classifier()

class RetrievePowersArgs(BaseModel):
    query: str = Field(..., description="提炼后的用户诉求核心事项，需要包含事项主体和事项内容（如：'街道路灯不亮'、'工地拖欠农民工工资'），必须明确且精炼。")
    top_k: int = Field(default=5, description="需要召回的权责清单数量，建议默认为 5。")


class RetrieveCasesArgs(BaseModel):
    query: str = Field(..., description="用户咨询的具体政策、办理条件或纠纷关键词（如：'53岁女性补缴养老保险'、'灵活就业公积金提取'）。")
    top_k: int = Field(default=5, description="需要召回的历史相似案例数量，建议默认为 5。")
    

class ClassifiyGovRequestArgs(BaseModel):
    query: str = Field(..., description="用户原始的完整问政诉求文本")


@tool("retrieve_powers_tool", args_schema=RetrievePowersArgs)
def retrieve_powers_tool(query: str, top_k: int = 5) -> str:
    """
    【权责清单与部门流转检索工具】
    核心用途：当你需要准确判断某个用户诉求（投诉、举报、求助）应该由哪个具体的政府部门（如：人社局、城管局、住建局）管辖时，必须调用此工具。
    返回值：包含相似度、相关的权责名称、行使主体（所属部门）以及具体的管理权限说明。
    请根据返回结果中的“负责部门”，指导用户或决定派单去向。
    """
    print(f"[Tool] 正在检索权责清单，提取管辖部门 | Query: {query}")
    
    docs = _powers_retriever.invoke(query)[:top_k]
    
    if not docs:
        return f"未找到与 '{query}' 相关的权责清单。请尝试提取更核心的业务关键词重新检索，或者建议派发综合工单。"
        
    results =[]
    for i, doc in enumerate(docs):
        results.append(f"【相似度】{doc.metadata['similarity']}，【匹配权责 {i+1}】{doc.page_content}")
        
    return "\n\n".join(results)


@tool("retrieve_cases_tool", args_schema=RetrieveCasesArgs)
def retrieve_cases_tool(query: str, top_k: int = 5) -> str:
    """
    【历史问政案例与政策检索工具】
    核心用途：当用户咨询具体的政务办理条件、政策细节、所需材料，或者你需要解答相似的业务问题时，必须调用此工具。
    返回值：过往相似的市民问政历史记录。包含相似度、历史问题及官方答复。
    """
    print(f"[Tool] 正在检索历史办理案例，提取政策依据 | Query: {query}")
    
    docs = _cases_retriever.invoke(query)[:top_k]
    
    if not docs:
        return f"未找到与 '{query}' 相关的历史办理案例。请尝试换一个相近的搜索词，或者告知用户暂无相关政策数据，询问是否需要转人工处理。"
        
    results =[]
    for i, doc in enumerate(docs):
        # 利用我们在底层检索器里存入的 QA Metadata
        results.append(f"【相似度】{doc.metadata['similarity']}，【相似案例 {i+1}】{doc.page_content}")
     
        
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
    print(f"[Tool] 正在对用户诉求进行意图分类 | Query: {query}")
    
    result: GovRequestClassifiedResult  = _gov_request_classifier.classify(query)
 
    req_type = result.request_type
    req_dept = result.request_department
    req_city_dept = result.request_city_department
    
    action_advice = ""
    
    if req_type == GovRequestType.CONSULT:
        action_advice = "当前意图为【咨询】。请立刻调用 `retrieve_cases_tool` 检索政策和历史类似问题的解答，为用户提供权威参考。"
    elif req_type in [GovRequestType.COMPLAINT, GovRequestType.ADVICE, GovRequestType.HELP]:
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