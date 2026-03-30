import re
import json
import numpy as np
import math
from loguru import logger

# 1. 从你的工具包中导入官方评测函数
from lmms_eval.tasks.SmolInstruct_v2.metrics import (
    calculate_smiles_metrics, 
    calculate_formula_metrics, 
    calculate_text_metrics, 
    calculate_number_metrics, 
    calculate_boolean_metrics
)
# 2. 导入你刚才封装的提取调度器
from lmms_eval.tasks.SmolInstruct_v2.prediction_extraction import extract_pred



TASK_PROMPTS = {
    # 属性预测 - 二分类 (HIV, BBBP, Clintox, Sider)
    "property_prediction-hiv": "Analyze the SMILES string and predict its HIV activity. Answer strictly with 'Yes' or 'No'.",
    "property_prediction-bbbp": "Does this molecule penetrate the Blood-Brain Barrier (BBBP)? Answer strictly with 'Yes' or 'No'.",
    "property_prediction-clintox": "Predict the clinical toxicity of this molecule. Answer strictly with 'Yes' or 'No'.",
    "property_prediction-sider": "Does this drug cause the specific side effect? Answer strictly with 'Yes' or 'No'.",
    
    # 属性预测 - 回归 (ESOL, Lipo)
    "property_prediction-esol": (
        "As a specialized chemist, calculate the logSol (aqueous solubility) for the molecule below. "
        "Consider the molecular weight, number of rotatable bonds, and aromatic proportion. "
        "Provide the final logSol value as a floating point number. Output: [value]"
    ),
    "property_prediction-lipo": (
        "Analyze the lipophilicity (logP) of this molecular structure. "
        "Focus on the hydrophobic and hydrophilic balance. "
        "Provide the numerical logP value only. Output: [value]"
    ),
    
    # 合成任务
    "forward_synthesis": "Predict the major product(s) for these reactants. Output the product(s) in SMILES format.",
    "retrosynthesis": "Suggest the starting materials (reactants) for the given product. If multiple reactants are needed, separate them with a period (.). Output only the SMILES strings.",
    
    # 格式/名称转换
    "name_conversion-s2i": (
        "Convert this SMILES structure to its IUPAC name. "
        "Output only the IUPAC name, without any explanation or extra text."
    ),

    "name_conversion-i2s": (
        "Convert this IUPAC name to its SMILES structure. "
        "Output only the SMILES string, without any explanation or extra text."
    ),

    "name_conversion-s2f": (
        "Determine the molecular formula of this SMILES string. "
        "Output only the molecular formula, without any explanation or extra text."
    ),

    "name_conversion-i2f": (
        "Determine the molecular formula of this IUPAC name. "
        "Output only the molecular formula, without any explanation or extra text."
    ),

    
    # 描述与生成
    "molecule_captioning": "Describe the following molecule's structure and chemical classification in detail:",
    "molecule_generation": "Generate the SMILES string for a molecule that fits this description:"
}

def extract_first_number(text):
    """从文本中提取第一个浮点数"""
    # 匹配包括负号和小数点的数字
    match = re.search(r"[-+]?\d*\.\d+|\d+", text)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return 0
    return 0

# ==========================================
# 1. lmms-eval 接口函数
# ==========================================

def smol_doc_to_visual(doc):
    return []

# def smol_doc_to_text(doc):
#     # 兼容 input (raw) 或 question (processed)
#     return doc.get("input", doc.get("question", ""))

def smol_doc_to_text(doc):
    """
    将 task-specific prompt + 原始输入 拼接成最终模型输入
    """
    task = doc["task"]
    user_input = doc.get("input", doc.get("question", "")).strip()

    prompt = TASK_PROMPTS.get(task, "").strip()

    if prompt:
        return f"{prompt}\n\n{user_input}"
    else:
        return user_input


def smol_doc_to_target(doc):
    # 【修复】优先读取 answer，如果不存在再读取 output
    return doc.get("answer", doc.get("output", ""))
def smol_process_results(doc, results):
    # 1. 安全获取原始输出
    raw_output = str(results[0]).strip() if results[0] is not None else ""
    task = doc["task"]
    MODEL_TYPE = "llama" 
    
    # --- 定义辅助函数：防截断的标签提取 ---
    def extract_between_tags_robust(text, tag_pattern):
        # 寻找开始标签 (例如 <SMILES>)
        match_start = re.search(tag_pattern, text, re.IGNORECASE)
        if match_start:
            content_after = text[match_start.end():]
            # 只要遇到 </ 就截断，不管后面是否完整
            close_idx = content_after.find('</')
            if close_idx != -1:
                return content_after[:close_idx].strip()
            else:
                return content_after.strip()
        return None

    # --- 2. 基础提取 (调用 extraction.py) ---
    temp_sample = {"output": [raw_output]}
    try:
        preds = extract_pred(temp_sample, MODEL_TYPE, task)
        # 确保不返回 None
        if preds and preds[0] is not None:
            pred = preds[0]
        else:
            pred = ""
    except Exception as e:
        logger.error(f"Extraction failed for task {task}: {e}")
        pred = raw_output 

    # --- 3. 定义任务分组 ---
    SMILES_TASKS = [
        'forward_synthesis', 'retrosynthesis', 
        'molecule_generation', 'name_conversion-i2s'
    ]
    FORMULA_TASKS = [
        'name_conversion-s2f', 'name_conversion-i2f'
    ]

    # --- 4. 核心 Filter 逻辑 (两种情况通吃) ---
    found_tag = False 

    # [情况 1]：尝试提取 XML 标签 (<SMILES> 或 <MOLFORMULA>)
    if task in SMILES_TASKS:
        extracted = extract_between_tags_robust(raw_output, r'<SMILES>|<SMILE>')
        if extracted:
            pred = extracted
            found_tag = True
            
    elif task in FORMULA_TASKS:
        # 兼容 <MOLFORMULA> 和 <Formula>
        extracted = extract_between_tags_robust(raw_output, r'<(?:MOL)?FORMULA>')
        if extracted:
            pred = extracted
            found_tag = True

    # [情况 2]：如果没有标签，且是自然语言句子，取最后一个词
    # 适用于 "This molecular's SMILES name is XXXXX"
    if not found_tag and (task in SMILES_TASKS or task in FORMULA_TASKS):
        if " " in pred:
            candidate = pred.split()[-1]
            
            # 清理句尾句号 (防止 "is XXXXX." 的情况)
            if candidate.endswith('.') and not re.search(r'\.[a-zA-Z0-9]', candidate):
                 candidate = candidate.rstrip('.')
            
            # 简单验证长度，避免提取到单个符号
            if len(candidate) > 1:
                pred = candidate

    # --- 5. 后处理 ---
    
    # A. 数值任务提取
    if task in ("property_prediction-esol", "property_prediction-lipo"):
        pred_num = extract_first_number(pred)
        pred = str(pred_num)
    
    # B. 分号替换 (SMILES 任务要求)
    if task in SMILES_TASKS and pred:
        # 移除残留的 XML 标签 (双重保险)
        pred = re.sub(r'<[^>]+>', '', pred).strip()
        pred = pred.replace(';', '.')

    # C. Captioning 任务防止 NoneType 报错
    if task == 'molecule_captioning':
        if pred is None:
            pred = ""
        else:
            pred = str(pred)

    return {
        "smol_metrics": {
            "pred": pred,
            "gold": doc["answer"],
            "task": task
        }
    }

# def smol_process_results(doc, results):
#     """
#     主要修改逻辑：
#     利用 extraction.py 里的函数从原始文本中抠出答案。
#     """
#     raw_output = str(results[0])
#     task = doc["task"]
    
#     MODEL_TYPE = "llama" 
    
#     # --- 1. 定义内部辅助函数 (防截断提取) ---
#     def extract_between_tags_robust(text, tag_pattern):
#         """
#         寻找开始标签，然后截取到最近的 '</' 为止。
#         tag_pattern: 例如 r'<(?:MOL)?FORMULA>' 匹配 <MOLFORMULA> 或 <Formula>
#         """
#         # 1. 寻找开始标签
#         match_start = re.search(tag_pattern, text, re.IGNORECASE)
#         if match_start:
#             # 2. 获取开始标签之后的所有内容
#             content_after = text[match_start.end():]
            
#             # 3. 寻找闭合标签的标志 '</'
#             # 只要遇到 </ 就认为结束了，不管后面跟的是 MOLFORMULA 还是 MOL...
#             close_idx = content_after.find('</')
            
#             if close_idx != -1:
#                 return content_after[:close_idx].strip()
#             else:
#                 # 如果完全没写闭合标签，就提取剩下全部
#                 return content_after.strip()
#         return None

#     # --- 2. 基础提取 (调用 extraction.py) ---
#     temp_sample = {"output": [raw_output]}
#     try:
#         preds = extract_pred(temp_sample, MODEL_TYPE, task)
#         pred = preds[0] if preds else ""
#     except Exception as e:
#         logger.error(f"Extraction failed for task {task}: {e}")
#         pred = raw_output.strip() 

#     if task in ("property_prediction-esol", "property_prediction-lipo"):
#         pred_num = extract_first_number(pred)
#         pred = str(pred_num)
    
#     # --- 3. 定义任务组 ---
#     TASKS_WITH_SEMICOLON_REPLACE = [
#         'forward_synthesis', 'retrosynthesis', 
#         'molecule_generation', 'name_conversion-i2s'
#     ]
#     # 【补全定义】
#     FORMULA_TASKS = [
#         'name_conversion-s2f', 'name_conversion-i2f'
#     ]

#     # --- 4. 标签清洗 Filter (覆盖 extract_pred 的结果) ---
    
#     # A. 处理 SMILES 任务的 <SMILES> 标签
#     if task in TASKS_WITH_SEMICOLON_REPLACE:
#         # 也可以改用 extract_between_tags_robust(raw_output, r'<SMILES>') 以防截断
#         match = re.search(r'<SMILES>\s*(.*?)\s*</SMILES>', raw_output, re.IGNORECASE | re.DOTALL)
#         if match:
#             pred = match.group(1).strip()
            
#     # B. 处理 Formula 任务的 <MOLFORMULA> 或 <Formula> 标签 (防截断)
#     elif task in FORMULA_TASKS:
#         extracted = extract_between_tags_robust(raw_output, r'<(?:MOL)?FORMULA>')
#         if extracted:
#             pred = extracted

#     # --- 5. 后处理 ---
#     if task in TASKS_WITH_SEMICOLON_REPLACE and pred:
#         pred = pred.replace(';', '.')


#     if task == 'molecule_captioning':
#         if pred is None:
#             pred = ""
#         else:
#             pred = str(pred)
        
#     return {
#         "smol_metrics": {
#             "pred": pred,
#             "gold": doc["answer"],
#             "task": task
#         }
#     }

# ==========================================
# 2. 指标汇总 (Aggregate)
# ==========================================

def is_valid_smiles(s):
    if not isinstance(s, str) or len(s.strip()) == 0:
        return False
    try:
        from rdkit import Chem
        return Chem.MolFromSmiles(s) is not None
    except:
        return False


def smol_aggregate_results(results):
    """
    收集所有样本的 pred 和 gold，统一调用官方提供的计算函数
    """
    task_groups = {}
    for res in results:
        # 这里 res 的 key 对应 smol_process_results 返回字典的 key
        data = res
        t = data["task"]
        if t not in task_groups:
            task_groups[t] = {"preds": [], "golds": []}
        
        # 官方 metric 期望格式是 List[List[str]] (为了支持单样本多候选)
        task_groups[t]["preds"].append([data["pred"]])
        task_groups[t]["golds"].append([data["gold"]])

    final_scores = {}
    score = 0.0 
    for task, data in task_groups.items():
        preds = data["preds"]
        golds = data["golds"]
        
        try:
            # 严格按照官方 main.py 中的分发逻辑选择计算函数
            if task in ('forward_synthesis', 'molecule_generation', 'name_conversion-i2s'):
                r = calculate_smiles_metrics(preds, golds)
                score = r.get('t1_morgan_fps', 0) # 使用 Morgan 相似度作为代表性指标
            elif task == 'retrosynthesis':
                valid_preds = [
                    p[0] for p in preds
                    if isinstance(p, list)
                    and len(p) > 0
                    and is_valid_smiles(p[0])
                ]

                if len(valid_preds) == 0:
                    score = 0.0
                else:
                    r = calculate_smiles_metrics(
                        preds, golds,
                        metrics=('exact_match', 'fingerprint', 'multiple_match')
                    )
                    score = r.get('t1_morgan_fps', 0)

            # elif task == 'retrosynthesis':
            #     valid_preds = [p for p in preds if is_valid_smiles(p)]

            #     if len(valid_preds) == 0:
            #         score = 0.0
            #     else:
            #         r = calculate_smiles_metrics(preds, golds, metrics=('exact_match', 'fingerprint', 'multiple_match'))
            #         score = r.get('t1_morgan_fps', 0) / len(valid_preds)
                # r = calculate_smiles_metrics(preds, golds, metrics=('exact_match', 'fingerprint', 'multiple_match'))
                # score = r.get('t1_morgan_fps', 0) / len(preds) if len(preds) > 0 else 0
            
            elif task == 'molecule_captioning':
                r = calculate_text_metrics(preds, golds)
                score = r.get('rouge_1', 0)
            
            elif task in ('name_conversion-i2f', 'name_conversion-s2f'):
                r = calculate_formula_metrics(preds, golds, metrics=('element_match',))
                score = r.get('num_t1_ele_match', 0) / len(preds) if len(preds) > 0 else 0
            
            elif task == 'name_conversion-s2i':
                r = calculate_formula_metrics(preds, golds, metrics=('split_match',))
                score = r.get('num_t1_split_match', 0) / len(preds) if len(preds) > 0 else 0
            
            elif task in ('property_prediction-esol', 'property_prediction-lipo'):
                r = calculate_number_metrics(preds, golds)
                rmse_val = r.get('RMSE', None)
                if rmse_val is None or not isinstance(rmse_val, (int, float)) or math.isnan(rmse_val):
                    score = 0.0
                else:
                    score = math.exp(-rmse_val)
            
            elif task in ('property_prediction-bbbp', 'property_prediction-clintox', 'property_prediction-hiv', 'property_prediction-sider'):
                r = calculate_boolean_metrics(preds, golds)
                score = r.get('num_correct', 0) / len(preds) if len(preds) > 0 else 0
            
            else:
                score = 0.0

            final_scores[f"{task}_score"] = score
        except Exception as e:
            logger.error(f"Error calculating metric for {task}: {e}")
            final_scores[f"{task}_score"] = 0.0
    return float(score)



