import os
import re
import sys
from pathlib import Path

import yaml
from loguru import logger as eval_logger

hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)

# 读取yaml文件
with open(Path(__file__).parent / "lsdbench.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


def lsdbench_doc_to_visual(doc):
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = doc["video_id"] + ".mp4"
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    elif os.path.exists(video_path.replace("mp4", "MP4")):
        video_path = video_path.replace("mp4", "MP4")
    elif os.path.exists(video_path.replace("mp4", "mkv")):
        video_path = video_path.replace("mp4", "mkv")
    else:
        # TODO: download video
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


def lsdbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    options = doc["options"]
    options_str = ""
    for option, option_text in options.items():
        options_str += f"{option}. {option_text}\n"
    context = question + "\n" + options_str
    return context


def extract_characters_regex(s):
    s = s.strip()

    pattern = r"(?<![a-zA-Z])[ABCD](?![a-zA-Z])"
    matches = re.findall(pattern, s)

    if matches:
        return matches[-1]  # return the last matched answer

    return ""


matrices = []


def lsdbench_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case videomme score), value: metric value
    """
    pred = results[0]
    pred_ans = extract_characters_regex(pred)
    gt_ans = doc["correct_answer"]
    video_id = doc["video_id"]

    data_dict = {"video_id": video_id, "pred_answer": pred_ans, "answer": gt_ans}

    return {f"accuracy": data_dict}


def lsdbench_aggregate_accuracy_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    total_correct = 0
    total_answered = 0

    for result in results:
        total_answered += 1
        total_correct += result["pred_answer"] == result["answer"]

    eval_logger.info(f"Overall Accuracy Performance: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")
    return 100 * total_correct / total_answered if total_answered > 0 else 0
