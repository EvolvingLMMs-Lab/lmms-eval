import datetime
import json
import os
from collections import defaultdict
from pathlib import Path
import yaml
from PIL import Image
import numpy as np
import random

from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

dir_name = os.path.dirname(os.path.abspath(__file__))

eval_type_dict = {
    "Subfield": [
        "Timbre",
        "Tone",
        "Melody",
        "Space",
        "Time",
        "Hallucination",
        "Intricacy",
    ],
}

hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)

question_prompt = "Answer with the option's letter from the given choices directly."


def av_odyssey_doc_to_visual(doc):
    visual_data = []
    
    # 获取当前目录的上一级目录
    base_dir = os.path.dirname(os.getcwd())
    base_dir = os.path.join(base_dir, 'AV_Odyssey_Bench_LMMs_Eval')
    # 处理 image 类型数据
    if 'image' in doc['data_type']:
        for relative_path in doc['image_path']:
            # 从上一级目录拼接路径
            relative_path = os.path.join(relative_path.split('/')[0], relative_path)
            abs_path = os.path.join(base_dir, relative_path)
            if os.path.exists(abs_path):
                visual_data.append(abs_path)  # 保留路径以供后续处理
            else:
                print(f"Image path does not exist: {abs_path}")
    
    # 处理 video 类型数据
    elif 'video' in doc['data_type']:
        for relative_path in doc['video_path']:
            # 从上一级目录拼接路径
            relative_path = os.path.join(relative_path.split('/')[0], relative_path)
            abs_path = os.path.join(base_dir, relative_path)
            if os.path.exists(abs_path):
                visual_data.append(abs_path)  # 保留路径以供后续处理
            else:
                print(f"Video path does not exist: {abs_path}")
                
    # 处理 audio 类型数据
    for relative_path in doc['audio_path']:
        # 从上一级目录拼接路径
        relative_path = os.path.join(relative_path.split('/')[0], relative_path)
        abs_path = os.path.join(base_dir, relative_path)
        if os.path.exists(abs_path):
            visual_data.append(abs_path)  # 保留路径以供后续处理
        else:
            print(f"Audio path does not exist: {abs_path}")

    return visual_data



def av_odyssey_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    options = doc["options"]
    option_text = options[0] + "\n" + options[1] + "\n" + options[2] + "\n"  + options[3] + "\n" 
    text = question + "\n" + option_text + question_prompt
    return text


def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    """
    for char in [',', '.', '!', '?', ';', ':', "'"]:
        response = response.strip(char)
    response = " " + response + " " # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f'{choice}' in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices: # e.g., A B C D
            if f' {choice} ' in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        # pred_index = random.choice(all_choices)
        pred_index = 'A'
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack: 
                for can in candidates:
                    index = response.rfind(f'({can})')
                    start_indexes.append(index) # -1 will be ignored anyway
                # start_indexes = [generated_response.index(f'({can})') for can in candidates]
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else: # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index



def av_odyssey_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case av_odyssey score), value: metric value
    """
    pred = results[0]
    options = doc["options"]
    option_list = {'A': options[0][3:], 'B': options[1][3:], 'C': options[2][3:], 'D': options[3][3:]}
    answer = parse_multi_choice_response(pred, ['A', 'B', 'C', 'D'], option_list)
    gt_answer = doc["answer"]
    assert answer in ["A", "B", "C", "D"]
    assert gt_answer in ["A", "B", "C", "D"]
    score = 1.0 if answer == gt_answer else 0.0
    category = doc["subfield"]
    key_name = "av_odyssey_score"
    # Note: the key name here is very important. It decides which aggregation function will receive the results
    # We note down the question id/category to help us aggregate the results later
    return {key_name: {"question_id": doc["question_id"], "category": category, "score": score}}




def av_odyssey_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    category2score = defaultdict(dict)
    for result in results:
        question_id = result["question_id"]
        score = result["score"]
        category = result["category"]
        if question_id not in category2score[category]:
            category2score[category][question_id] = []
        category2score[category][question_id].append(score)
    
    # 计算每个 category 的平均分
    category_avg_scores = {}
    total_score = 0
    total_questions = 0

    # 遍历所有 category 来计算每个 category 的平均分
    for category, questions in category2score.items():
        # import pdb
        # pdb.set_trace()
        category_total = 0  # 计算所有问题的总分
        for question_id, score in questions.items():
            category_total += score[0]  # 累加所有问题的平均分
        category_avg_scores[category] = category_total / len(questions) * 100.0  # 当前类别的平均分

        total_score += category_total  # 累加所有类别的问题总分
        total_questions += len(questions)  # 累加所有问题的数量

    # 计算所有问题的平均分（按问题的总数来平均）
    overall_avg_score = total_score / total_questions * 100.0

    # 输出每个 category 的平均分
    print("Average scores per category:")
    for category, avg_score in category_avg_scores.items():
        print(f"{category}: {avg_score:.2f}")

    # 输出所有问题的平均分
    print(f"Overall average score (across all questions): {overall_avg_score:.2f}")
    
    return overall_avg_score