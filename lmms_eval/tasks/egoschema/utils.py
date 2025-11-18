import datetime
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import yaml
from decord import VideoReader, cpu

import lmms_eval.tasks._task_utils.file_utils as file_utils

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
HF_HOME = os.environ["HF_HOME"] if "HF_HOME" in os.environ else os.path.expanduser("~/.cache/huggingface/hub")
cache_dir = config["dataset_kwargs"]["cache_dir"]
cache_dir = os.path.join(HF_HOME, cache_dir)
cache_dir = os.path.join(cache_dir, "videos")

from loguru import logger as eval_logger

from PIL import Image as PIL_Image

def egoschema_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    visuals = egoschema_doc_to_visual(doc)
    if visuals is None:
        visuals = []
    text = egoschema_doc_to_text(doc, lmms_eval_specific_kwargs=lmms_eval_specific_kwargs)
    messages = [{"role": "user", "content": []}]
    content = []
    for visual in visuals:
        if isinstance(visual, PIL_Image.Image):
            content.append({"type": "image", "url": visual})
        elif isinstance(visual, dict):
            content.append({"type": "audio", "url": visual})
        elif isinstance(visual, str):
            content.append({"type": "video", "url": visual, "question": egoschema_doc_to_question(doc)})
    content.append({"type": "text", "text": text})
    messages[0]["content"] = content
    return messages


# Pass in video path here
# Can only work correctly with video llm
def egoschema_doc_to_visual(doc):
    video_path = doc["video_idx"] + ".mp4"
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    elif os.path.exists(video_path.replace("mp4", "MP4")):
        video_path = video_path.replace("mp4", "MP4")
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]

def egoschema_doc_to_question(doc):
    return doc["question"]

# This is the place where you format your question
def egoschema_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    if "post_prompt" in lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs["post_prompt"]

    question = egoschema_doc_to_question(doc)
    if "option" in doc:
        for op in doc["option"]:
            question += "\n" + op
        post_prompt = "\nAnswer with the option's letter from the given choices directly."

    return f"{pre_prompt}{question}{post_prompt}"


def egoschema_doc_to_answer(doc):
    return doc["answer"]


# Process result for mc_ppl
def egoschema_process_results(doc, result):
    # Initialize minimum value and index
    min_value = float("inf")
    min_index = -1

    # Iterate through the results to find the index of the lowest value
    for i, (value, _) in enumerate(result):
        if value < min_value:
            min_value = value
            min_index = i

    # Return the result with the index of the lowest value
    return {"submission": {doc["video_idx"]: min_index}, "score": {"pred": min_index, "ground_truth": doc["answer"]}}


def get_multi_choice_info(doc):
    all_choices = []
    index2ans = {}
    OPTIONS = ["A", "B", "C", "D", "E"]
    for i in range(5):
        # import pdb;pdb.set_trace()
        index2ans[OPTIONS[i]] = doc["option"][i].strip()
        all_choices.append(OPTIONS[i])

    return index2ans, all_choices


def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L10
    """
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    ans_with_space = False
    ans_with_dot = False
    candidates = []
    # import pdb; pdb.set_trace()
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            candidates.append(f"({choice})")
            ans_with_brack = True

    # if len(candidates) == 0:
    for choice in all_choices:  # e.g., A B C D
        if f"{choice} " in response:
            candidates.append(f"{choice} ")
            ans_with_space = True

    # if len(candidates) == 0:
    for choice in all_choices:  # e.g., A. B. C. D.
        if f"{choice}." in response:
            candidates.append(f"{choice}.")
            ans_with_dot = True

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        # candidates = list(set(candidates))
        start_indexes = []
        if index_ans:
            # if ans_with_brack:
            for can in candidates:
                index = response.rfind(can)
                start_indexes.append(index)  # -1 will be ignored anyway
                # start_indexes = [generated_response.index(f'({can})') for can in candidates]
            # if ans_with_space:
            #     for can in candidates:
            #         index = response.rfind(f"{can} ")
            #         start_indexes.append(index)
            # if ans_with_dot:
            #     for can in candidates:
            #         index = response.rfind(f"{can}.")
            #         start_indexes.append(index)
            # if not ans_with_brack and not ans_with_space and not ans_with_dot:
            #     for can in candidates:
            #         index = response.rfind(f" {can} ")
            #         start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the first one
        pred_index = candidates[np.argmin(start_indexes)]
        pred_index = pred_index.replace("(", "").replace(")", "").replace(".", "").strip()
    else:  # if only one candidate, use it.
        pred_index = candidates[0]
        pred_index = pred_index.replace("(", "").replace(")", "").replace(".", "").strip()

    return pred_index, len(candidates) > 0


# Process result for mcq answer generation
def egoschema_process_results_generation(doc, result):
    # import pdb;pdb.set_trace()
    pred = result[0]

    index2ans, all_choices = get_multi_choice_info(doc)
    parsed_pred, matched_tag = parse_multi_choice_response(pred, all_choices, index2ans)

    pred_to_index = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    index = pred_to_index.get(parsed_pred, -1)  # Default to -1 if the prediction is not found

    return {"submission": {doc["video_idx"]: index}, "score": {"pred": index, "ground_truth": doc["answer"]}}


def egoschema_aggregate_submissions(results, args, task):
    now_date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    submission_file_name = f"inference_results_egoschema_{task}_{now_date_time}.json"
    path = file_utils.generate_submission_file(submission_file_name, args)

    # results is a list of 5031 dict,
    # need to convert results into a single dict with 5031 key-value pairs
    combined_submission = {}

    for submission_dict in results:
        combined_submission.update(submission_dict)

    with open(path, "w") as f:
        json.dump(combined_submission, f, indent=4)

    eval_logger.info(f"Submission file saved to {path}")


# Factory into different aggregate
def egoschema_aggregate_mc(results, args):
    egoschema_aggregate_submissions(results, args, "MC")


def egoschema_aggregate_mc_ppl(results, args):
    egoschema_aggregate_submissions(results, args, "MC_PPL")


def egoschema_aggregate_score(results, args):
    yes_count = 0

    # results is a list of dict
    for answer_dict in results:
        if str(answer_dict["ground_truth"]) == str(answer_dict["pred"]):
            yes_count = yes_count + 1

    accuracy = yes_count / len(results)

    return accuracy


def egoschema_doc_to_choice(doc):
    return [op.split(".")[1].strip() for op in doc["option"]]
