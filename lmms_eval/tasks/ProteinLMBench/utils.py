import re

def doc_to_visual(doc):
    return []

def doc_to_text(doc):
    # 官方定义的标准 Prompt
    instruction = (
        "Answer the multiple-choice question based solely on the provided context. \n"
        "If you are still unsure about the answer, output option 7.\n"
        "Select only ONE correct option by its number. Start your response with 'The correct option is' followed by the option number ONLY. eg: \"The correct option is Option X.\"\n"
    )
    
    question = doc["question"]
    # 拼接选项
    options_str = ""
    for option in doc["options"]:
        options_str += option + "\n"
    
    # 按照官方格式组合
    full_prompt = f"{instruction}\n Question: \n{question}\n Options: \n{options_str}\nThe correct option is:"
    return full_prompt

def doc_to_target(doc):
    """
    从 "option 1" 这种格式中提取出数字 "1" 作为目标答案。
    """
    # 强制转为 string 防止报错
    answer_str = str(doc.get("answer", ""))
    match = re.search(r'\d+', answer_str)
    return match.group() if match else ""

def process_results(doc, results):
    """
    解析模型生成的文本，提取选项数字并对比。
    """
    # lmms-eval 传入的 results 是一个列表，取出第一个生成结果
    # 加上 strip() 去除首尾空白，加上 str() 防止结果不是字符串
    prediction_text = str(results[0]).strip() if results else ""
    
    pred_num = "None"
    
    # 尝试寻找模型输出中的第一个数字
    try:
        # 因为 Prompt 强制了 "The correct option is:" 结尾，
        # 所以模型生成的第一个数字通常就是答案。
        pred_match = re.search(r'\d+', prediction_text)
        if pred_match:
            pred_num = pred_match.group()
    except Exception:
        pred_num = "None"

    # 获取标准答案数字
    target = doc_to_target(doc)
    
    # 计算准确率 (强制转字符串对比，避免 "1" != 1 的情况)
    score = 1.0 if str(pred_num) == str(target) else 0.0
    
    return {
        "protein_lm_accuracy": score,
        "extracted_answer": pred_num,
        "ground_truth": target
    }