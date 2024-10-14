import datetime
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

DISCIPLINES = ["Tech & Engineering", "Science", "Health & Medicine", "Sports & Arts", "Game", "Business", "Embodied Tasks"]


replace_prompt = " Please answer yes or no."

# with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
#     raw_data = f.readlines()
#     safe_data = []
#     for i, line in enumerate(raw_data):
#         # remove function definition since yaml load cannot handle it
#         if "!function" not in line:
#             safe_data.append(line)

#     config = yaml.safe_load("".join(safe_data))

hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
# cache_dir = os.path.join(hf_home, cache_dir)
# base_cache_dir = config["dataset_kwargs"]["cache_dir"]
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "mmworld.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


def extract_and_remove_subfolders(cache_dir):
    # Walk through all the subdirectories and move files to the root of cache_dir
    for root, dirs, files in os.walk(cache_dir):
        for file in files:
            source = os.path.join(root, file)
            destination = os.path.join(cache_dir, file)
            if source != destination:
                shutil.move(source, destination)

    for root, dirs, files in os.walk(cache_dir, topdown=False):
        for dir in dirs:
            os.rmdir(os.path.join(root, dir))


def mmworld_doc_to_visual(doc):
    cache_dir = os.path.join(base_cache_dir, cache_name)
    extract_and_remove_subfolders(cache_dir)
    video_path_doc = doc["video_id"].split("/")[-1] + ".mp4"
    video_path = os.path.join(cache_dir, video_path_doc).replace(".mp4.mp4", ".mp4")

    if os.path.exists(video_path):
        video_path = video_path
    elif os.path.exists(video_path.replace("mp4", "MP4")):
        video_path = video_path.replace("mp4", "MP4")
    elif os.path.exists(video_path.replace("mp4", "avi")):
        video_path = video_path.replace("mp4", "avi")
    elif os.path.exists(os.path.join(cache_dir, "shorts:" + video_path_doc)):
        video_path = os.path.join(cache_dir, "shorts:" + video_path_doc)
    elif os.path.exists(os.path.join(cache_dir, "shorts:" + doc["video_id"].split("/")[-1] + ".MP4")):
        video_path = os.path.join(cache_dir, "shorts:" + doc["video_id"].split("/")[-1] + doc["video_id"] + ".MP4")
    elif os.path.exists(os.path.join(cache_dir, "shorts:" + doc["video_id"].split("/")[-1] + ".avi")):
        video_path = os.path.join(cache_dir, "shorts:" + doc["video_id"].split("/")[-1] + ".avi")
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")

    return [video_path]


def mmworld_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    option_prompt = "Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option."
    question = doc["question"]
    option = str(doc["options"])
    question = question + "\n" + option
    post_prompt = lmms_eval_specific_kwargs["post_prompt"] if "post_prompt" in lmms_eval_specific_kwargs else "The best answer is:"
    full_prompt = option_prompt + "\n" + question + "\n" + post_prompt
    return full_prompt


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is" "The correct option is",
        "Best answer:" "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""

    matches = re.search(r"[ABCD]", s)
    if matches is None:
        return ""
    return matches[0]


def mmworld_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case videomme score), value: metric value
    """
    pred = results[0]
    pred_ans = extract_characters_regex(pred)
    # gt_ans = doc["answer"].lower().strip().replace(".", "")

    discipline = doc["discipline"]
    data_dict = {"video_id": doc["video_id"], "discipline": discipline, "pred_answer": pred_ans, "answer": doc["correct_answer_label"].upper()}

    return {f"mmworld_accuracy": data_dict}


def mmworld_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    category2score = {}

    for category in DISCIPLINES:
        key = f"{category}"
        category2score[key] = {"correct": 0, "answered": 0}

    for result in results:
        category = result["discipline"]
        key = f"{category}"
        category2score[key]["answered"] += 1
        category2score[key]["correct"] += result["pred_answer"] == result["answer"]

    for category in DISCIPLINES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if category in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        eval_logger.info(f"Evaluation on DISCIPLINES: {category}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    total_correct = 0
    total_answered = 0
    for k, v in category2score.items():
        total_correct += v["correct"]
        total_answered += v["answered"]
    eval_logger.info(f"Overall Performance: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")
    return 100 * total_correct / total_answered if total_answered > 0 else 0
