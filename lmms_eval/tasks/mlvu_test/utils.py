import datetime
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union
import pdb
import cv2
import numpy as np
import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

# TASK_TYPES = ["TR", "AR", "VS", "NQA", "ER", "PQA", "SSC", "AO", "AC"]

TASK_TYPES = {'anomaly_reco',
 'count',
 'ego',
 'needleQA',
 'order',
 'plotQA',
 'sportsQA',
 'topic_reasoning',
 'tutorialQA'}


# hf_home = os.getenv("HF_HOME", "./~/.cache/huggingface")
hf_home="/share/junjie/shuyan/lmms-eval/~/.cache/huggingface"
base_cache_dir = os.path.expanduser(hf_home)

with open(Path(__file__).parent / "mlvu_test.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


def mlvu_doc_to_visual(doc):
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = doc["video_name"]
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


def mlvu_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    # option_prompt="Carefully watch this video and pay attention to every detail. Based on your observations, select the best option that accurately addresses the question."
    option_prompt = ""
    try:
        question = doc["question"] + "\nOnly give the best option.\n"
    except:
        pdb.set_trace()
    full_prompt = option_prompt + "\n" + question + "\n" + "Best option: ("
    return full_prompt


def extract_characters_regex(s):
    s = s.strip()
    if ")" in s:
        index = s.index(")")
        pred = s[index - 1 : index]
        return pred
    else:
        return s

# def extract_characters_regex(s):
#     s = s.strip()
#     answer_prefixes = [
#         "The best answer is",
#         "The correct answer is",
#         "The answer is",
#         "The answer",
#         "The best option is" "The correct option is",
#         "Best answer:" "Best option:",
#     ]
#     for answer_prefix in answer_prefixes:
#         s = s.replace(answer_prefix, "")

#     if len(s.split()) > 10 and not re.search("[ABCD]", s):
#         return ""

#     matches = re.search(r"[ABCD]", s)
#     if matches is None:
#         return ""
#     return matches[0]

def mlvu_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case videomme score), value: metric value
    """
    pred = results[0]
    # print("****************",pred)
    pred_ans = extract_characters_regex(pred)

    task_type = doc["task_type"]
    data_dict = {"question_id": doc["question"], "task_type": task_type, "pred_answer": pred_ans, "answer": doc["answer"]}

    return {f"mlvu_perception_score": data_dict}


def mlvu_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    category2score = {}
    for task_type in TASK_TYPES:
        category2score[task_type] = {"correct": 0, "answered": 0}

    for result in results:
        task_type = result["task_type"]
        category2score[task_type]["answered"] += 1
        category2score[task_type]["correct"] += result["pred_answer"] == result["answer"]

    task_category_scores = {}

    # Calculate and log accuracy for each task category
    for task_cate in TASK_TYPES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if task_cate in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        accuracy = 100 * total_correct / total_answered if total_answered > 0 else 0
        task_category_scores[task_cate] = accuracy
        eval_logger.info(f"Evaluation on Task Categories: {task_cate}: {accuracy:.1f}%")

    # Calculate and log average accuracy across all task categories
    if TASK_TYPES:
        average_accuracy = sum(task_category_scores.values()) / len(TASK_TYPES)
    else:
        average_accuracy = 0

    eval_logger.info(f"Average Performance Across All Task Categories: {average_accuracy:.1f}%")

    return average_accuracy
