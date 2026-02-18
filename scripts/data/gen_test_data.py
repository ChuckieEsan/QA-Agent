import os
import sys
import pandas as pd
import json
import hashlib
import time
from tqdm import tqdm
import dashscope
from dashscope import Generation

sys.path.append(os.getcwd())

from config.setting import settings
from app.infra.utils import generate_doc_id

DATA_PATH = str(settings.RAW_DATA_PATH)
OUTPUT_PATH = str(settings.QUERY_TEST_DATA_PATH)
dashscope.api_key = settings.QWEN_API_KEY

SAMPLE_SIZE = 300
DEPT_TOP_K = 60


def load_and_sample_data():
    """读取并进行加权抽样 (Weighted Sampling)"""
    df = pd.read_excel(DATA_PATH)
    
    # 1. 列名映射
    rename_map = {}
    for col in df.columns:
        if "问政内容" in col: rename_map[col] = "question"
        elif "回复单位" in col: rename_map[col] = "department"
        elif "回复内容" in col: rename_map[col] = "answer"
    
    df = df.rename(columns=rename_map).dropna(subset=['question', 'answer'])
    
    # 2. 生成 doc_id
    print("正在计算 MD5 ID 以匹配数据库...")
    df['doc_id'] = df.apply(lambda row: generate_doc_id(row['question'], row['answer']), axis=1)
    
    # 3. 筛选 Top K 部门
    top_depts = df['department'].value_counts().head(DEPT_TOP_K).index
    df_filtered = df[df['department'].isin(top_depts)].copy()
    
    print(f"正在进行加权采样 (样本池大小: {len(df_filtered)})...")

    # 4. 计算采样权重
    # -------------------------------------------------------------
    # 策略：平方根平滑加权 (Square Root Smoothing)
    # 公式：Weight = sqrt(Department_Count)
    # -------------------------------------------------------------
    # 为什么这样做？
    # 假设 Dept_A 有 10000 条，Dept_B 有 100 条。
    # - 纯随机/按比例采样 (100:1)：Dept_A 占绝大多数，Dept_B 可能一条都抽不到。
    # - 等额采样 (1:1)：Dept_A 和 Dept_B 都是 5 条，无法体现 Dept_A 的热门程度。
    # - 平方根采样 (100:10 = 10:1)：既保证了热门部门样本更多，又保护了小部门的存在感。
    
    # a. 获取每个部门在当前池子中的总数
    dept_counts = df_filtered['department'].value_counts()
    
    # b. 映射权重 (Pandas 的 sample 会自动归一化权重，无需手动除以总数)
    weights = df_filtered['department'].map(lambda x: dept_counts[x] ** 0.5)
    
    # 5. 执行加权抽样
    # replace=False: 确保不会抽到重复的文档 (前提是样本池 > SAMPLE_SIZE)
    # random_state=42: 保证每次运行脚本抽到的测试集是一样的，便于复现
    if len(df_filtered) >= SAMPLE_SIZE:
        sampled_df = df_filtered.sample(n=SAMPLE_SIZE, weights=weights, random_state=42, replace=False)
    else:
        print(f"样本池数量 ({len(df_filtered)}) 少于目标采样数 ({SAMPLE_SIZE})，将全量返回。")
        sampled_df = df_filtered
        
    print(f"采样完成。部门分布示例:\n{sampled_df['department'].value_counts().head(5)}")
    return sampled_df


def rewrite_query(original_text, max_retries=3):
    """
    调用大模型生成 3 个不同风格的测试问题 (包含重试机制)
    :param max_retries: 最大重试次数，默认为 3
    """
    short_text = str(original_text)
    
    # Prompt 保持不变
    prompt = f"""
    你是一个政务数据测试助手。请基于“市民问政内容”，模拟真实用户，生成 3 个不同风格的搜索 Query。
    
    原内容：{short_text}
    
    请严格按照以下 JSON 格式返回列表，不要包含 Markdown 标记或其他废话：
    ["简短关键词风格", "口语化提问风格", "稍微复杂/啰嗦的风格"]
    """
    
    last_usage = None # 用于记录最后一次尝试的 token 消耗
    
    for attempt in range(max_retries):
        try:
            # 发起调用
            resp = Generation.call(
                model=Generation.Models.qwen_turbo, 
                prompt=prompt,
                result_format='message' 
            )
            
            # 记录 Usage (即使失败也可能有消耗，尽量保留)
            if resp.usage:
                last_usage = resp.usage

            # 1. 检查 API 状态码
            if resp.status_code == 200:
                content = resp.output.choices[0].message.content
                # 清洗 Markdown (Qwen 偶尔会带 ```json)
                content = content.replace("```json", "").replace("```", "").strip()
                
                try:
                    # 2. 尝试解析 JSON
                    queries = json.loads(content)
                    
                    # 3. 校验数据结构 (必须是列表且不为空)
                    if isinstance(queries, list) and len(queries) > 0:
                        return queries, last_usage
                    else:
                        print(f"[第 {attempt+1} 次] 格式错误 (非列表): {content[:30]}...")
                
                except json.JSONDecodeError:
                    print(f"[第 {attempt+1} 次] JSON 解析失败: {content[:30]}...")
            else:
                # API 层面报错 (如限流、服务器错误)
                print(f"[第 {attempt+1} 次] API 报错 ({resp.code}): {resp.message}")
        
        except Exception as e:
            # 网络或其他未知异常
            print(f"[第 {attempt+1} 次] 发生异常: {e}")
        
        # 如果还没到最后一次尝试，则进行等待 (指数退避: 1s, 2s, 4s...)
        if attempt < max_retries - 1:
            sleep_time = 2 ** attempt
            time.sleep(sleep_time)
            
    # === 所有重试都失败后的降级处理 ===
    print(f"重试 {max_retries} 次后仍失败，降级使用原始文本。")
    return [short_text], last_usage


def main():
    print(f"正在构建测试集 (Target: {OUTPUT_PATH})...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    sampled_df = load_and_sample_data()
    print(f"已抽取 {len(sampled_df)} 条样本，开始 LLM 改写...")
    
    results = []
    
    # === 初始化 Token 计数器 ===
    total_input_tokens = 0
    total_output_tokens = 0
    
    for _, row in tqdm(sampled_df.iterrows(), total=len(sampled_df)):
        original_q = str(row['question'])
        
        # 这里返回的是一个列表 queries_list
        queries_list, usage = rewrite_query(original_q)
        
        # === 累加 Token 消耗 ===
        if usage:
            total_input_tokens += usage.input_tokens
            total_output_tokens += usage.output_tokens
        
        # 将返回的 3 个问题拆成 3 条独立的测试数据
        for variant in queries_list:
            entry = {
                "doc_id": row['doc_id'],            
                "original_query": original_q,       
                "test_query": variant,
                "department": str(row.get('department')),
                "answer": str(row.get('answer', ''))
            }
            results.append(entry)
        
    # 写入 JSONL
    print(f"正在写入 JSONL 文件...")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print(f"测试集生成完毕！共 {len(results)} 行。")
    if results:
        # 打印前3条来看看（对应同一条原始数据的3个变体）
        print(f"查看数据示例:\n{json.dumps(results[:3], ensure_ascii=False, indent=2)}")
    
    # === 输出统计报告 ===
    print("\n" + "="*30)
    print("本次生成消耗统计")
    print("="*30)
    print(f"Total Input Tokens : {total_input_tokens}")
    print(f"Total Output Tokens: {total_output_tokens}")
    print(f"Total Tokens       : {total_input_tokens + total_output_tokens}")
    print("="*30)

if __name__ == "__main__":
    main()