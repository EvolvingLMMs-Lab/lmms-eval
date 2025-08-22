import json
import logging
import os
from pathlib import Path

eval_logger = logging.getLogger(__name__)

# Semantic match prompts from eval.py in step2_audio_paralinguistic
SEMANTIC_MATCH_PROMPTS = {
    "default": """请评估以下两个文本是否描述了相似的内容。

文本1: {text1}
文本2: {text2}

只需回答小写的"yes"或"no"，不要解释:""",

    "gender": """请评估以下两个文本中是否都提到了相同性别的描述（"男"或"女"）。

文本1: {text1}
文本2: {text2}

判断标准：
1. 如果两个文本都包含"男"或男性相关词汇（如"男人"、"男性"、"男声"等），回答"yes"
2. 如果两个文本都包含"女"或女性相关词汇（如"女人"、"女性"、"女声"等），回答"yes"
3. 如果一个文本提到"男"而另一个提到"女"，回答"no"

只需回答小写的"yes"或"no"，不要解释:""",

    "speed": """请评估以下两个文本描述的语速级别是否相同或相邻。
文本1: {text1}
文本2: {text2}

语速级别定义（从快到慢）：
1. 快速
2. 中快速
3. 中速
4. 中慢速
5. 慢速

判断规则：
- 如果两个描述属于同一级别 → "yes"
- 如果相差一个级别（如"快速"和"中快速"） → "yes"
- 如果相差两个或更多级别 → "no"
- 如果无法确定具体级别 → "no"

只需回答小写的"yes"或"no"，不要解释:""",

    "voice_tone": """请评估以下两个文本中描述说话人的音色是否大体上相似。

文本1: {text1}
文本2: {text2}

只需回答小写的"yes"或"no"，不要解释:""",

    "rhythm": """请评估以下两个文本中描述说话人的节奏是否大体相似。

文本1: {text1}
文本2: {text2}

宽松匹配规则：
1. 只要双方都提到"平稳"、"流畅"、"自然"中的任意一个词，就认为匹配
2. 只要双方都提到"停顿"(无论何种停顿)，就认为匹配
3. "急促"和"波动"只要双方都有速度/节奏变化的描述就认为匹配

只需回答小写的"yes"或"no"，不要解释:""",

    "voice_styles": """请评估以下两个文本中描述说话人的语音风格是否大体上相似。

文本1: {text1}
文本2: {text2}

只需回答小写的"yes"或"no"，不要解释:""",

    "pitch": """请评估以下两个文本中描述说话人的音调是否大致相同。

文本1: {text1}
文本2: {text2}

只需回答小写的"yes"或"no"，不要解释:""",

    "emotions": """请评估以下两个文本描述的情感是否属于相近类别。
文本1: {text1}
文本2: {text2}

**情感分类及匹配规则：**
1. **积极类**（可互相匹配）：
   - 高兴/兴奋/愉快/热情/自豪/得意/温柔/撒娇/喜爱
2. **中性类**（可互相匹配）：
   - 平静/中性/冷静/客观/怀旧/沉思/稳重
3. **消极类**（可互相匹配）：
   - 愤怒/不满/沮丧/无奈/烦躁/指责/嘲讽/轻蔑/委屈/焦虑/绝望/痛苦/恐惧/羞愧

只需回答小写的 "yes" 或 "no"，不要解释：""",
    
    "scene": """请判断以下两个文本描述的音频场景是否一致：
规则：
1. 允许表述差异（如「在厨房」和「厨房里的声音」算匹配）。
2. 忽略无关符号或修饰词（如「<中文>」「的声音」「录的」等）。
3. 户外与公园要做区分，不属于同一个场景。

文本1: {text1}
文本2: {text2}

只需回答小写的 "yes" 或 "no",不要解释：""",
    
    "age": """请评估以下两个文本描述的说话人年龄范围是否相似（允许±10岁误差）。

文本1: {text1}
文本2: {text2}

判断步骤：
1. 提取文本1中的最小和最大年龄（如"20-30岁"→20和30）
2. 提取文本2中的最小和最大年龄
3. 计算两组年龄的中点
4. 如果两个中点相差≤10岁，回答"yes"；否则"no"

只需回答小写的"yes"或"no"，不要解释:""",
    
    "event": """请判断以下两个文本描述的声音事件是否在以下任一情况下匹配：
1. 描述同类事件（如都是动物声音、交通工具声等）
2. 语义上存在关联（如"歌声"和"音乐"）

文本1: {text1}
文本2: {text2}

只需回答小写的"yes"或"no":""",
    
    "vocalsound": """请判断以下两段文本中描述的声音/行为是否属于以下同类情况：
1. 相同类型的声音行为（如"咳嗽"和"咳嗽声"）
2. 相同情绪表达（如"笑声"和"笑声"）
3. 包含关系（如"咳嗽声"和"咳嗽和清嗓子的声音"）

文本1: {text1}
文本2: {text2}

根据以上标准，只需回答小写的"yes"或"no":"""
}

def doc_to_audio(doc):
    """Extract audio path from document"""
    return [doc["audio"]]

def doc_to_text(doc, lmms_eval_specific_kwargs):
    """Generate text prompt based on task type"""
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    
    task_name = doc["task_name"]
    
    prompts = {
        "识别说话人年龄": "请根据音频中说话人的声音特征，判断说话人的年龄范围。",
        "识别说话人情绪": "请根据音频中说话人的语调和语气，描述说话人的情绪状态。",
        "识别说话人性别": "请根据音频中说话人的声音特征，判断说话人的性别。",
        "识别音频场景": "请根据音频内容，识别音频的场景环境。",
        "识别语音事件": "请根据音频内容，识别音频中发生的声音事件。",
        "识别说话人音调": "请根据音频中说话人的声音，描述说话人的音调特征。",
        "识别说话人语速": "请根据音频中说话人的语速，描述说话人的语速特点。",
        "识别说话人节奏": "请根据音频中说话人的说话方式，描述说话人的语音节奏。",
        "识别说话人声音风格": "请根据音频中说话人的声音，描述说话人的声音风格特征。",
        "识别说话人音色": "请根据音频中说话人的声音，描述说话人的音色特征。",
        "识别语音行为": "请根据音频内容，识别音频中的语音行为或声音类型。"
    }
    
    prompt = prompts.get(task_name, "请分析这段音频。")
    
    return f"{pre_prompt}{prompt}{post_prompt}"

def doc_to_target(doc):
    """Extract target answer from document"""
    return doc["task_answer"]

def process_results(doc, result):
    """Process model results and compare with ground truth"""
    pred = result[0] if len(result) > 0 else ""
    gt = doc["task_answer"]
    
    task_type = doc["subset"]
    
    audio_path = ""
    if "audio" in doc:
        if isinstance(doc["audio"], dict):
            audio_path = doc["audio"].get("path", "")
        else:
            audio_path = str(doc["audio"])
    else:
        eval_logger.debug(f"Available keys in doc: {list(doc.keys())}")
        audio_path = "unknown"
    
    return {
        "semantic_match": {
            "pred": pred,
            "gt": gt,
            "task_type": task_type,
            "audio_path": audio_path
        }
    }

def judge_semantic_match(answer, asr_text, prompt_template):
    """
    Use OPENAI LLM to judge semantic consistency
    """
    try:
        from openai import OpenAI
        
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        formatted_prompt = prompt_template.format(text1=answer, text2=asr_text)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一个专业的文本评估助手"},
                {"role": "user", "content": formatted_prompt}
            ],
            temperature=0
        )
        
        result = response.choices[0].message.content.strip().lower()
        return 1 if result == "yes" else 0
        
    except ImportError:
        eval_logger.error("OpenAI library not found. Install with: pip install openai")
        return 0
    except Exception as e:
        eval_logger.error(f"Error in semantic matching: {e}")
        return 0

def semantic_match_aggregate(results, args=None):
    """Aggregate semantic matching results using eval.py logic"""
    
    results_by_task = {}
    for result in results:
        task_type = result["task_type"]
        if task_type not in results_by_task:
            results_by_task[task_type] = []
        results_by_task[task_type].append(result)
    
    task_accuracies = {}
    overall_correct = 0
    overall_total = 0
    
    for task_type, task_results in results_by_task.items():
        correct = 0
        total = len(task_results)
        
        prompt_template = SEMANTIC_MATCH_PROMPTS.get(task_type, SEMANTIC_MATCH_PROMPTS["default"])
        
        for result in task_results:
            try:
                match = judge_semantic_match(result["gt"], result["pred"], prompt_template)
                correct += match
            except Exception as e:
                eval_logger.error(f"Error evaluating semantic match: {e}")
                pass
        
        accuracy = correct / total if total > 0 else 0
        task_accuracies[task_type] = accuracy
        
        overall_correct += correct
        overall_total += total
        
        eval_logger.info(f"Task {task_type}: {correct}/{total} = {accuracy:.4f}")
    
    overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0
    eval_logger.info(f"Overall accuracy: {overall_correct}/{overall_total} = {overall_accuracy:.4f}")
    
    return overall_accuracy
