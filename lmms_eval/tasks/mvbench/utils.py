import datetime
import json
import os
import re
import string
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import PIL
import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

DATA_LIST = {
    "object_interaction": "star/Charades_segment",
    "action_sequence": "star/Charades_segment",
    "action_prediction": "star/Charades_segment",
    "action_localization": "sta/sta_video_segment",
    "moving_count": "clevrer/video_validation",
    "fine_grained_pose": "nturgbd_convert",
    "character_order": "perception/videos",
    "object_shuffle": "perception/videos",
    "egocentric_navigation": "vlnqa",
    "moving_direction": "clevrer/video_validation",
    "episodic_reasoning": "tvqa/video_fps3_hq_segment",
    "fine_grained_action": "Moments_in_Time_Raw/videos",
    "scene_transition": "scene_qa/video",
    "state_change": "perception/videos",
    "moving_attribute": "clevrer/video_validation",
    "action_antonym": "ssv2_video_mp4",
    "unexpected_action": "FunQA_test/test",
    "counterfactual_inference": "clevrer/video_validation",
    "object_existence": "clevrer/video_validation",
    "action_count": "perception/videos",
}

hf_home = os.getenv("HF_HOME", "~/.cache/huggingface")
base_cache_dir = os.path.expanduser(hf_home)

with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)


cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


def mvbench_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    cache_dir = os.path.join(base_cache_dir, cache_name)
    dataset_folder = DATA_LIST[lmms_eval_specific_kwargs["sub_task"]]
    video_path = os.path.join(cache_dir, dataset_folder, doc["video"])
    if os.path.exists(video_path):
        video_path = video_path
    elif os.path.basename(dataset_folder) in ["clevrer", "star"]:
        alternative_video_path = os.path.join(cache_dir, "data0613", dataset_folder, doc["video"])
        if os.path.exists(alternative_video_path):
            video_path = alternative_video_path
        else:
            eval_logger.error(f"Video path: {video_path} does not exist, please check.")
    else:
        eval_logger.error(f"Video path: {video_path} does not exist, please check.")
    return [video_path]


def mvbench_frames_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    cache_dir = os.path.join(base_cache_dir, cache_name)
    dataset_folder = DATA_LIST[lmms_eval_specific_kwargs["sub_task"]]
    video_path = os.path.join(cache_dir, dataset_folder, doc["video"])
    if os.path.exists(video_path):
        video_path = video_path
    elif os.path.basename(dataset_folder) in ["clevrer", "star"]:
        alternative_video_path = os.path.join(cache_dir, "data0613", dataset_folder, doc["video"])
        if os.path.exists(alternative_video_path):
            video_path = alternative_video_path
        else:
            eval_logger.error(f"Video path: {video_path} does not exist, please check.")
    else:
        eval_logger.error(f"Video path: {video_path} does not exist, please check.")

    frame_path_list = [os.path.join(video_path, f) for f in os.listdir(video_path) if f.endswith(".jpg") or f.endswith(".png")]
    frame_image_list = [PIL.Image.open(frame_path).convert("RGB") for frame_path in frame_path_list]
    return frame_image_list


def mvbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    option_prompt = ""
    option_list = doc["candidates"]
    option_letters = string.ascii_uppercase
    for char_index, option in enumerate(option_list):
        option_letter = option_letters[char_index]
        option_prompt += f"{option_letter}. {option}\n"

    full_text = "Question:" + doc["question"] + "\nOption:\n" + option_prompt + lmms_eval_specific_kwargs["post_prompt"]
    return full_text


def mcq_acc(answer, pred):
    periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
    commaStrip = re.compile("(\d)(\,)(\d)")
    punct = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]

    def processPunctuation(inText):
        outText = inText
        for p in punct:
            if (p + " " in inText or " " + p in inText) or (re.search(commaStrip, inText) != None):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = periodStrip.sub("", outText, re.UNICODE)
        return outText

    def process(answer):
        option_regex = re.compile(r"^([A-E])\.\s*(.+)$", re.IGNORECASE)
        match = option_regex.match(answer.strip())

        if match:
            # If matched, return the option letter in uppercase
            return match.group(1).upper()
        else:
            # If no match, process the answer as before
            answer = answer.replace("\n", " ")
            answer = answer.replace("\t", " ")
            answer = answer.strip()
            answer = processPunctuation(answer)
            answer = answer.strip("'")
            answer = answer.strip('"')
            answer = answer.strip(")")
            answer = answer.strip("(")
            answer = answer.strip().lower()

            # Try to find any single letter (A-E) in the processed answer
            letter_match = re.search(r"\b([A-E])\b", answer, re.IGNORECASE)
            if letter_match:
                return letter_match.group(1).upper()

            return answer

    pred = process(pred)
    answer = process(answer)

    if pred == answer:
        score = 1
    else:
        score = 0

    return score


def mvbench_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case mvbench_perception_score), value: metric value
    """
    pred = results[0]

    # Calculate the ground truth option letter
    option_letters = string.ascii_uppercase
    gt_option_letter = None
    for i, candidate in enumerate(doc["candidates"]):
        if candidate == doc["answer"]:
            gt_option_letter = option_letters[i]
            break

    # Calculate the score using mcq_acc function
    score = mcq_acc(gt_option_letter, pred)

    data_dict = {"pred_answer": pred, "gt_answer": gt_option_letter, "score": score}

    return {"mvbench_accuracy": data_dict}


def mvbench_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    total_answered = 0
    total_correct = 0
    for result in results:
        if result["pred_answer"] != "":
            total_answered += 1
            total_correct += result["score"]

    return 100 * total_correct / total_answered if total_answered > 0 else 0
