from decord import VideoReader, cpu
import numpy as np
import os
import sys
import datetime
import lmms_eval.tasks._task_utils.file_utils as file_utils
import json
import logging
import yaml
from pathlib import Path

with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

# We will unzip all the zip files
# To HF HOME cache dir
# And load it here
HF_HOME = os.environ["HF_HOME"]
cache_dir = config["dataset_kwargs"]["cache_dir"]
cache_dir = os.path.join(HF_HOME, cache_dir)
cache_dir = os.path.join(cache_dir, "videos")

eval_logger = logging.getLogger("lmms-eval")


# Pass in video path here
# Can only work correctly with video llm
def perceptiontest_val_doc_to_visual(doc):
    video_path = doc["video_name"] + ".mp4"
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    elif os.path.exists(video_path.replace("mp4", "MP4")):
        video_path = video_path.replace("mp4", "MP4")
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


# This is the place where you format your question
def perceptiontest_val_doc_to_text(doc, model_specific_prompt_kwargs=None):
    if model_specific_prompt_kwargs is None:
        model_specific_prompt_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in model_specific_prompt_kwargs:
        pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
    if "post_prompt" in model_specific_prompt_kwargs:
        post_prompt = model_specific_prompt_kwargs["post_prompt"]

    question = doc["question"]
    if "options" in doc:
        index = 0
        for op in doc["options"]:
            if index == 0:
                question += "\n" + "A. " + op
            elif index == 1:
                question += "\n" + "B. " + op
            else:
                question += "\n" + "C. " + op
            index += 1
        post_prompt = "\nAnswer with the option's letter from the given choices directly."

    return f"{pre_prompt}{question}{post_prompt}"


def perceptiontest_val_doc_to_answer(doc):
    return doc["answer_id"]


# Process result for mc_ppl
def perceptiontest_val_process_results_mc_ppl(doc, result):
    # Initialize minimum value and index
    min_value = float("inf")
    min_index = -1

    # Iterate through the results to find the index of the lowest value
    for i, (value, _) in enumerate(result):
        if value < min_value:
            min_value = value
            min_index = i

    # Return the result with the index of the lowest value
    return {
        "accuracy": {
            "video_name": doc["video_name"],
            "question": doc["question"],
            "question_id": doc["question_id"],
            "pred_id": min_index,
            "answer_id": doc["answer_id"],
            "area": doc["area"],
            "reasoning": doc["reasoning"],
            "tag": doc["tag"],
        }
    }


# Process result for generation
def perceptiontest_val_process_results_mc(doc, result):
    pred = result[0]  # string prediction "A", "B", "C"

    # Map the prediction to an index
    pred_to_index = {"A": 0, "B": 1, "C": 2}
    index = pred_to_index.get(pred, -1)  # Default to -1 if the prediction is not found

    return {
        "accuracy": {"video_name": doc["video_name"], "question": doc["question"], "question_id": doc["question_id"], "pred_id": index, "answer_id": doc["answer_id"], "area": doc["area"], "reasoning": doc["reasoning"], "tag": doc["tag"]}
    }


def perceptiontest_val_aggregate_accuracy(results, args):
    yes_count = 0

    # results is a list of dict
    for answer_dict in results:
        if str(answer_dict["answer_id"]) == str(answer_dict["pred_id"]):
            yes_count = yes_count + 1

    accuracy = yes_count / len(results)

    return accuracy


def perceptiontest_val_doc_to_choice(doc):
    return [op for op in doc["options"]]
