import os
import sys

sys.path.append(os.getcwd())

from src.app.infra.llm.default_llm_service import DefaultLLMService
from src.app.components.classifier.base_classifier import (
    GovRequestClassifiedResult,
    GovRequestType,
    GovRequestUrgency
)
from src.app.components.classifier.request_classifier import GovRequestClassifier


if __name__ == "__main__":
#     service = DefaultLLMService(provider="deepseek", model_name="deepseek-chat")
    
#     # response = service.generate([
#     #     {"role": "system", "content": "你是一个政务助手，专门回答政府相关政策问题，请提供准确的政策信息。"},
#     #     {"role": "user", "content": "什么是'雨露计划'？申请条件是什么？"}
#     # ])
#     messages = [
#         {"role": "system", "content": """你是政务问政分类专家。请对以下市民诉求进行分类，并判断紧急程度。

# 分类标准：
# 1. 建议（advice）：对政府工作提出改进建议、意见
# 2. 投诉（complaint）：反映政府部门或工作人员的问题、不当行为
# 3. 求助（help）：请求政府帮助解决个人或家庭困难
# 4. 咨询（consult）：询问政策、流程、办事指南等信息
# 5. 其他（other）：与政务无关的内容

# 紧急程度：
# - major：紧急（涉及生命财产安全、重大民生问题）
# - normal：一般（常规政务事项）
# - minor：轻微（咨询、建议类）

# 请以JSON格式输出，包含以下字段：
# - request_type: 分类类型（advice/complaint/help/consult/other）
# - request_urgency: 紧急程度（major/normal/minor）"""},
#             {"role": "user", "content": "建议在公交站台增设座椅，方便老年人等车时休息。"}
#         ]
    
#     response = service.generate_structured(messages, GovRequestClassifiedResult, temperature=0)
    classifier = GovRequestClassifier()
    response = classifier.classify("建议在公交站台增设座椅，方便老年人等车时休息。")
    print(response)
    