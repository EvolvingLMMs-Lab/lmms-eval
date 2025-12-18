# Important Note:
# Videos have been sourced from three different recognized datasets: SoccerNet, MovieChat-1K, and Walking_Tours.
# While Walking_Tours videos are already in the repo, the following steps are required to download the remaining videos:
#
# 1. Fill out the Non-Disclosure Agreement form for the SoccerNet dataset:
#    https://docs.google.com/forms/d/e/1FAIpQLSfYFqjZNm4IgwGnyJXDPk2Ko_lZcbVtYX73w5lf6din5nxfmA/viewform
#    Save the password (SOCCERNET_PWD) sent to your email in the os.
#
# 2. Request access to the MovieChat-1K dataset:
#    https://huggingface.co/datasets/Enxin/MovieChat-1K_train
#
# The script will prompt you to enter the SoccerNet password during execution.

import ast
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import requests
import seaborn as sns
import torch
import yaml
from decord import VideoReader, cpu
from loguru import logger as eval_logger
from matplotlib import pyplot as plt
from PIL import Image

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

SOCCERNET_PWD = os.getenv("SOCCERNET_PWD", "s0cc3rn3t")

NUM_SECONDS_TO_SLEEP = 5

GPT_EVAL_MODEL_NAME = "gpt-4o-mini"

API_TYPE = os.getenv("API_TYPE", "openai")

if API_TYPE == "openai":
    API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    API_KEY = os.getenv("OPENAI_API_KEY", "")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

hf_home = os.getenv("HF_HOME", "~/.cache/huggingface")
base_cache_dir = os.path.expanduser(hf_home)
benchmark_cache_dir = os.path.join(base_cache_dir, "hub", "datasets--cplou99--FALCON-Bench", "snapshots")


with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)
if "dataset_kwargs" in yaml.safe_load("".join(safe_data)) and "cache_dir" in yaml.safe_load("".join(safe_data))["dataset_kwargs"]:
    cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]
else:
    cache_name = None


def download_and_organize_data(cache_dir):
    # Run the script with arguments
    print(
        """
        Please ensure you have completed the following steps before running this script:
        1. Fill out the Non-Disclosure Agreement form for the SoccerNet dataset:
        https://docs.google.com/forms/d/e/1FAIpQLSfYFqjZNm4IgwGnyJXDPk2Ko_lZcbVtYX73w5lf6din5nxfmA/viewform
        Save the password (SOCCERNET_PWD) sent to your email in the environment variable SOCCERNET_PWD.
        2. Request access to the MovieChat-1K dataset:
        https://huggingface.co/datasets/Enxin/MovieChat-1K_train
    """
    )
    snapshots = sorted(os.listdir(benchmark_cache_dir))
    snapshot_benchmark_cache_dir = os.path.join(benchmark_cache_dir, snapshots[-1])
    script_path = os.path.join(snapshot_benchmark_cache_dir, "download_videos.py")
    result = subprocess.run(["python", script_path, "--soccernet_password", SOCCERNET_PWD, "--hf_benchmark_dir", snapshot_benchmark_cache_dir, "--data_dir", cache_dir], stdout=None, stderr=None)


def move_key_first(d, key):
    if key in d:
        d = {key: d[key], **{k: v for k, v in d.items() if k != key}}
    return d


def timestamp_to_seconds(timestamp):
    # Split the timestamp into hours, minutes, and seconds
    h, m, s = timestamp.split(":")
    # Convert hours, minutes, and total seconds (including fractions) to float and compute total seconds
    total_seconds = int(h) * 3600 + int(m) * 60 + float(s)
    return total_seconds


def load_video(video_file, duration, max_num_frames=16):
    from decord import VideoReader

    vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
    fps = vr.get_avg_fps()
    total_valid_frames = len(vr)
    num_frames = min(max_num_frames, int(total_valid_frames))

    frame_indices = [int(total_valid_frames / num_frames) * i for i in range(num_frames)]

    frames = vr.get_batch(frame_indices)
    if isinstance(frames, torch.Tensor):
        frames = frames.numpy()
    else:
        frames = frames.asnumpy()
    frame_timestamps = [frame_index / fps for frame_index in frame_indices]

    return [Image.fromarray(fr).convert("RGB") for fr in frames]


def compute_frame_timestamps(duration, max_num_frames=16):
    if duration > max_num_frames:
        return [duration / max_num_frames * i for i in range(max_num_frames)]
    else:
        return [i for i in range(int(duration))]


def FALCONbench_doc_to_choice(doc):
    return doc["options"]


def FALCONbench_doc_to_target(doc):
    return doc["gt_answer"]


def FALCONbench_doc_to_text_mcq(doc, lmms_eval_specific_kwargs):
    candidates = []

    question = doc["question"] + "\n" + "\n".join([". ".join([chr(ord("A") + i), candidate]) for i, candidate in enumerate(doc["options"])])
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = "Answer with the option's letter from the given choices directly"

    return f"{pre_prompt}\n{question}\n{post_prompt}"


def FALCONbench_doc_to_text_mcq_temploc(doc, lmms_eval_specific_kwargs):
    candidates = []

    question = doc["question"] + "\n" + "\n".join([". ".join([chr(ord("A") + i), candidate]) for i, candidate in enumerate(doc["options"])])
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = f"""\n **Output Format Instructions:**
    You must provide your final answer strictly as a JSON object enclosed in a markdown code block.

    The JSON object must contain exactly these two keys:
    1. "response": Answer with the option's letter from the given choices directly.
    2. "temporal_window": A list of two integers [start_second, end_second] representing the time interval in seconds where the answer is observed.

    **Example of required output:**
    ```json
    {{
    "response": "A person running",
    "temporal_window": [105, 140]
    }}
    ```
    """
    return f"{pre_prompt}\n{question}\n{post_prompt}"


def FALCONbench_doc_to_text_oq(doc, lmms_eval_specific_kwargs):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]

    return f"{pre_prompt}\n{question}\n{post_prompt}"


def FALCONbench_doc_to_text_oq_temploc(doc, lmms_eval_specific_kwargs):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = f"""\n **Output Format Instructions:**
    You must provide your final answer strictly as a JSON object enclosed in a markdown code block.

    The JSON object must contain exactly these two keys:
    1. "response": Answer with the option's letter from the given choices directly.
    2. "temporal_window": A list of two integers [start_second, end_second] representing the time interval in seconds where the answer is observed.

    **Example of required output:**
    ```json
    {{
    "response": "A person running",
    "temporal_window": [105, 140]
    }}
    ```
    """
    return f"{pre_prompt}\n{question}\n{post_prompt}"


hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)


def FALCONbench_doc_to_visual_fullvideo(doc):
    if cache_name is None:
        cache_dir = os.path.join(base_cache_dir, "hub", "datasets--cplou99--FALCON-Bench")
    else:
        cache_dir = os.path.join(base_cache_dir, cache_name)
    video_name = doc["path"]
    full_videos_path = os.path.join(cache_dir, "full_videos")
    if not os.path.exists(full_videos_path):
        print("Let us start downloading and organizing the full videos")
        download_and_organize_data(cache_dir)

    video_path = os.path.join(full_videos_path, video_name)
    if os.path.exists(video_path):
        video_path = video_path
        # print(f"video path:{video_path} exists")
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


def get_multi_choice_info(options):
    """
    Given the list of options for multiple choice question
    Return the index2ans and all_choices
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/data_utils.py#L54
    """

    start_chr = "A"
    all_choices = []
    index2ans = {}
    for i, option in enumerate(options):
        index2ans[chr(ord(start_chr) + i)] = option
        all_choices.append(chr(ord(start_chr) + i))

    return index2ans, all_choices


def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Changed from MMMU-style complex parsing into simple parsing.
    Fixed to avoid 'D. A book' be parsed as A.
    Same as original FALCONbench paper, if parsing failed, it will assign a random choice to model.
    """
    s = response.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return random.choice(all_choices)

    matches = re.search(r"[ABCD]", s)
    if matches is None:
        for letter, option in index2ans.items():
            if option.lower() in s.lower():
                return letter
        return random.choice(all_choices)
    return matches[0]


def evaluate_FALCONbench(samples):
    pred_correct = 0
    judge_dict = dict()
    for sample in samples:
        gold_i = sample["answer"]
        pred_i = sample["parsed_pred"]
        correct = eval_multi_choice(gold_i, pred_i)

        if correct:
            judge_dict[sample["id"]] = "Correct"
            pred_correct += 1
        else:
            judge_dict[sample["id"]] = "Wrong"

    if len(samples) == 0:
        return {"acc": 0}
    return judge_dict, {"acc": pred_correct / len(samples)}


def eval_multi_choice(gold_i, pred_i):
    correct = False
    # only they are exactly the same, we consider it as correct
    if isinstance(gold_i, list):
        for answer in gold_i:
            if answer == pred_i:
                correct = True
                break
    else:  # gold_i is a string
        if gold_i == pred_i:
            correct = True
    return correct


def evaluate_FALCONbench_gpteval(results):
    score, acc = 0, 0
    for result in results:
        eval_score = result["score"]
        try:
            eval_score = int(eval_score)
        except:
            eval_score = 0.0
        score += eval_score

        eval_acc = result["acc"]
        try:
            eval_acc = str(eval_acc)
            if eval_acc == "yes":
                acc += 1
        except:
            acc += 0
    return {"score": score / len(results), "acc": acc / len(results)}


def calculate_ins_level_acc(results):
    """Calculate the instruction level accuracy for given Subject results
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L246
    """
    acc = 0
    ins_num = 0
    accs = []
    for cat_results in results.values():
        acc += cat_results["acc"] * cat_results["num_example"]
        ins_num += cat_results["num_example"]
        accs.append(cat_results["acc"])
    if ins_num == 0:
        return 0
    final_acc = acc / ins_num
    avg_acc = sum(accs) / len(accs)
    return final_acc, avg_acc


def calculate_ins_level_acc_score(results):
    """Calculate the instruction level accuracy for given Subject results
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L246
    """
    acc = 0
    score = 0
    ins_num = 0
    accs, scores = [], []
    for cat_results in results.values():
        acc += cat_results["acc"] * cat_results["num_example"]
        score += cat_results["score"] * cat_results["num_example"]
        ins_num += cat_results["num_example"]
        accs.append(cat_results["acc"])
        scores.append(cat_results["score"])
    if ins_num == 0:
        return 0
    return {"acc": acc / ins_num, "score": score / ins_num, "avg_acc": sum(accs) / len(accs), "avg_score": sum(scores) / len(scores)}


def get_eval_generic(question, answer, pred, max_tokens: int, retries: int = 5):
    global headers

    messages = [
        {
            "role": "system",
            "content": "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
            "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
            "------"
            "##INSTRUCTIONS: "
            "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
            "- Consider synonyms or paraphrases as valid matches.\n"
            "- Evaluate the correctness of the prediction compared to the answer.",
        },
        {
            "role": "user",
            "content": "Please evaluate the following video-based question-answer pair:\n\n"
            f"Question: {question}\n"
            f"Correct Answer: {answer}\n"
            f"Predicted Answer: {pred}\n\n"
            "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
            "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
            "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}.",
        },
    ]

    payload = {
        "model": GPT_EVAL_MODEL_NAME,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        # "response_format": {"type": "json_object"},
    }

    for attempt in range(retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()  # Raises HTTPError for bad responses
            try:
                response_data = response.json()  # Attempt to parse JSON
            except requests.exceptions.JSONDecodeError:
                eval_logger.error(f"JSON decode error on attempt {attempt + 1}. Response text: {response.text}")
                continue  # Skip to next retry
            content = response_data["choices"][0]["message"]["content"].strip()
            if content != "":
                return content, response_data["model"]
        # Handle HTTP errors separately
        except requests.exceptions.HTTPError as e:
            eval_logger.error(f"HTTP error on attempt {attempt + 1}: {e}")
        # Handle other requests-related errors
        except requests.exceptions.RequestException as e:
            eval_logger.error(f"Request exception on attempt {attempt + 1}: {e}")
        except Exception as e:
            eval_logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")

        if "Sorry! We've encountered an issue with repetitive patterns in your prompt. Please try again with a different prompt." in json.loads(response.content)["error"]["message"]:
            eval_logger.error(f"Repetitive patterns in prompt. Drop this data.")
            return "", ""

        # Handle other unexpected errors
        if attempt < retries - 1:
            time.sleep(NUM_SECONDS_TO_SLEEP)
        else:  # If this was the last attempt, log and return empty
            eval_logger.error(f"All {retries} attempts failed.")
            return "", ""

    return "", ""


def parse_score(review):
    try:
        # Convert the string representation of a dictionary to an actual dictionary
        review_dict = ast.literal_eval(review)
        score = review_dict.get("score", 0)
        return int(score)
    except SyntaxError as e:
        eval_logger.error(f"Syntax error parsing the review string: {e}. Review content: {review}")
        return 0
    except ValueError as e:
        eval_logger.error(f"Value error parsing the review string: {e}. Review content: {review}")
        return 0
    except Exception as e:
        eval_logger.error(f"Unexpected error parsing the review string: {e}. Review content: {review}")
        return 0


def parse_acc(review):
    try:
        # Convert the string representation of a dictionary to an actual dictionary
        review_dict = ast.literal_eval(review)
        pred = review_dict.get("pred", "no")
        return str(pred)
    except SyntaxError as e:
        eval_logger.error(f"Syntax error parsing the review string: {e}. Review content: {review}")
        return "no"
    except ValueError as e:
        eval_logger.error(f"Value error parsing the review string: {e}. Review content: {review}")
        return "no"
    except Exception as e:
        eval_logger.error(f"Unexpected error parsing the review string: {e}. Review content: {review}")
        return "no"


def gpt_eval(data_dict):
    evaluated_results = []

    try:
        question = data_dict["question"]
        answer = data_dict["gt_answer"]
        pred = data_dict["pred"]

        # Assume get_eval returns a review and the model name, and parse_score parses this review
        review, model_name = get_eval_generic(question, answer, pred, 64)
        score = parse_score(review)
        acc = parse_acc(review)
    except Exception as e:
        eval_logger.error(f"Error for Video Name: {data_dict.get('video_name', 'Unknown')}: {e}")
        review = "Failed to Get a Proper Review."
        model_name = ""
        score = 0
        acc = "no"

    # Update the dictionary with the new entries
    updated_dict = {
        "question_id": data_dict["question_id"],
        "review": review,
        "score": score,
        "acc": acc,
    }

    return updated_dict


def evaluate_temporal_localization(pred_interval, gt_interval):
    if pred_interval is None or gt_interval is None:
        temp_loc_dict = {"IoU": 0, "GToU": 0, "pred_interval_length": 0, "gt_interval_length": 0}
        return temp_loc_dict
    start_gt, end_gt = gt_interval[0], gt_interval[1]
    start_pred, end_pred = pred_interval[0], pred_interval[1]

    # Calculate the length of the ground truth interval (GT)
    gt_length = max(0, end_gt - start_gt)
    pred_length = max(0, end_pred - start_pred)

    # Calculate the union of GT and Pred
    union_start = min(start_gt, start_pred)
    union_end = max(end_gt, end_pred)
    union_length = max(0, union_end - union_start)

    # Calculate the intersection of GT and Pred
    intersection_start = max(start_gt, start_pred)
    intersection_end = min(end_gt, end_pred)
    intersection_length = max(0, intersection_end - intersection_start)

    # Compute the GToU metric
    if intersection_length > 0:
        GToU = (gt_length / union_length) * 1  # The indicator is implicitly 1 if there's an overlap
    elif start_gt <= start_pred <= end_pred <= end_gt:
        GToU = 1
    else:
        GToU = 0  # No overlap, so GToU is 0

    if union_length > 0:
        IoU = intersection_length / union_length
    else:
        IoU = 0

    temp_loc_dict = {"IoU": round(IoU, 5), "GToU": round(GToU, 5), "pred_interval_length": round(pred_length, 5), "gt_interval_length": round(gt_length, 5)}
    return temp_loc_dict


def evaluate_FALCONbench_temporal_localization(results):
    mIoU, mGToU, mPredLength, mGTLength = 0, 0, 0, 0
    for result in results:
        mIoU += result["temp_loc"]["IoU"]
        mGToU += result["temp_loc"]["GToU"]
        mPredLength += result["temp_loc"]["pred_interval_length"]
        mGTLength += result["temp_loc"]["gt_interval_length"]
    if len(results) == 0:
        temp_loc_dict = {"mIoU": 0, "mGToU": 0, "mPredLength": 0, "mGTLength": 0}
    else:
        temp_loc_dict = {"mIoU": round(mIoU / len(results), 5), "mGToU": round(mGToU / len(results), 5), "mPredLength": round(mPredLength / len(results), 5), "mGTLength": round(mGTLength / len(results), 5)}
    return temp_loc_dict


def calculate_ins_level_temporal_localization(results):
    mIoU, mGToU, mPredLength, mGTLength = 0, 0, 0, 0
    avg_mIoU, avg_mGToU, avg_mPredLength, avg_mGTLength = 0, 0, 0, 0
    num_samples = 0
    num_splits = 0
    for cat_results in results.values():
        mIoU += cat_results["temp_loc"]["mIoU"] * cat_results["num_example"]
        mGToU += cat_results["temp_loc"]["mGToU"] * cat_results["num_example"]
        mPredLength += cat_results["temp_loc"]["mPredLength"] * cat_results["num_example"]
        mGTLength += cat_results["temp_loc"]["mGTLength"] * cat_results["num_example"]
        avg_mIoU += cat_results["temp_loc"]["mIoU"]
        avg_mGToU += cat_results["temp_loc"]["mGToU"]
        avg_mPredLength += cat_results["temp_loc"]["mPredLength"]
        avg_mGTLength += cat_results["temp_loc"]["mGTLength"]
        num_samples += cat_results["num_example"]
        num_splits += 1

    temporal_localization_results = {
        "mIoU": round(mIoU / num_samples, 5),
        "mGToU": round(mGToU / num_samples, 5),
        "mPredLength": round(mPredLength / num_samples, 5),
        "mGTLength": round(mGTLength / num_samples, 5),
        "avg_mIoU": round(avg_mIoU / num_splits, 5),
        "avg_mGToU": round(avg_mGToU / num_splits, 5),
        "avg_mPredLength": round(avg_mPredLength / num_splits, 5),
        "avg_mGTLength": round(avg_mGTLength / num_splits, 5),
    }
    return temporal_localization_results


def _parse_temporal_window(value):
    """
    Helper to convert various inputs into a valid list of two floats/ints.
    Returns None if conversion fails.
    """
    if value is None:
        return None

    # Step A: If it's a string, try to turn it into a list
    if isinstance(value, str):
        value = value.strip()
        # Handle simple cases like "10, 20" by adding brackets
        if not value.startswith("["):
            value = f"[{value}]"

        try:
            # ast.literal_eval is safer than eval()
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            print(f"Failed to parse temporal_window string: {value}")
            return None

    # Step B: Now checks if it is a list
    if not isinstance(value, list):
        print(f"temporal_window is not a list: {value}")
        return None

    # Step C: Check length and content types
    if len(value) != 2:
        print(f"temporal_window does not have exactly two elements: {value}")
        return None

    # Ensure items are numbers (int or float)
    if not all(isinstance(x, (int, float)) for x in value):
        print(f"temporal_window elements are not all numbers: {value}")
        return None

    return value


def parse_json_response(output_text):
    """
    Parses VLM output to extract 'response' and 'temporal_window'.
    Robustly handles cases where 'temporal_window' is returned as a string
    (e.g., "[10, 20]") instead of a list.
    """
    default_dict_response = {"response": "None", "temporal_window": None}
    # 1. Regex to find the JSON block (or first valid curly brace structure)
    pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
    match = re.search(pattern, output_text, re.DOTALL)

    json_str = None
    if match:
        json_str = match.group(1)
    else:
        # Fallback: manually find the first { and last }
        start = output_text.find("{")
        end = output_text.rfind("}") + 1
        if start != -1 and end != 0:
            json_str = output_text[start:end]

    if not json_str:
        print(f"Failed to find JSON block in output: {output_text}")
        return default_dict_response

    try:
        # 2. Initial Parse
        data = json.loads(json_str)
        if not isinstance(data, dict):
            print(f"Parsed JSON is not a dictionary: {data}")
            return default_dict_response

        # 3. Validate 'response'
        if "response" not in data or not isinstance(data["response"], str):
            print(f"Invalid or missing 'response' in parsed data: {data}")
            data["response"] = "None"

        # 4. Validate and Fix 'temporal_window'
        raw_window = data.get("temporal_window")
        data["temporal_window"] = _parse_temporal_window(raw_window)

        return data

    except json.JSONDecodeError:
        return default_dict_response


def FALCONbench_process_results_mcq(doc, results):
    pred = results[0]

    all_choices = []
    index2ans = {}
    for i in range(len(doc["options"])):
        option = doc["options"][i]
        if option == "N/A":
            break
        index2ans[chr(ord("A") + i)] = option
        all_choices.append(chr(ord("A") + i))

    parsed_pred = parse_multi_choice_response(pred, all_choices, index2ans)
    id = doc["question_id"]
    gt_option = [chr(ord("A") + i) for i in range(len(doc["options"]))][doc["gt_option_idx"]]
    acc = {"id": id, "dataset": doc["dataset"], "category": doc["category"], "answer": gt_option, "parsed_pred": parsed_pred}

    return {
        "acc": acc,
        "submission": {
            id: pred,
        },
    }


def FALCONbench_process_results_mcq_temploc(doc, results):
    if type(results[0]) == str:
        pred_dict = parse_json_response(results[0])
    elif type(results[0]) == dict and "response" in results[0] and "temporal_window" in results[0]:
        pred_dict = results[0]
    else:
        raise ValueError(f"Invalid prediction format for question_id {doc['question_id']}. The output must be a string or a dictionary with 'response' and 'temporal_window' keys.")

    pred = pred_dict["response"]
    all_choices = []
    index2ans = {}
    for i in range(len(doc["options"])):
        option = doc["options"][i]
        if option == "N/A":
            break
        index2ans[chr(ord("A") + i)] = option
        all_choices.append(chr(ord("A") + i))

    parsed_pred = parse_multi_choice_response(pred, all_choices, index2ans)
    id = doc["question_id"]
    gt_option = [chr(ord("A") + i) for i in range(len(doc["options"]))][doc["gt_option_idx"]]
    acc = {"id": id, "dataset": doc["dataset"], "category": doc["category"], "answer": gt_option, "parsed_pred": parsed_pred}

    acc["pred_dict"] = pred_dict
    if "temporal_window" in pred_dict:
        temp_loc_dict = evaluate_temporal_localization(pred_dict["temporal_window"], doc["gt_time_interval"])
        acc["temp_loc"] = temp_loc_dict
    else:
        raise ValueError(f"No temporal_window found in prediction for question_id {id}. The output must be a dictionary with 'response' and 'temporal_window' keys.")
    return {
        "acc": acc,
        "submission": {
            id: pred,
        },
    }


# Process result for evaluation in generic task
def FALCONbench_process_results_oq(doc, result):
    pred = result[0]
    doc["pred"] = pred
    eval_results = gpt_eval(doc)
    result_dict = {
        "gpt_eval_score_acc": {
            "question_id": doc["question_id"],
            "dataset": doc["dataset"],
            "category": doc["category"],
            "question": doc["question"],
            "answer": doc["gt_answer"],
            "pred": pred,
            "score": eval_results["score"],
            "acc": eval_results["acc"],
            "review": eval_results["review"],
        }
    }
    return result_dict


# Process result for evaluation in generic task
def FALCONbench_process_results_oq_temploc(doc, result):
    if type(result[0]) == str:
        pred_dict = parse_json_response(result[0])
    elif type(result[0]) == dict and "response" in result[0] and "temporal_window" in result[0]:
        pred_dict = result[0]
    else:
        raise ValueError(f"Invalid prediction format for question_id {doc['question_id']}. The output must be a string or a dictionary with 'response' and 'temporal_window' keys.")

    pred = pred_dict["response"]
    doc["pred"] = pred
    eval_results = gpt_eval(doc)
    result_dict = {
        "gpt_eval_score_acc": {
            "question_id": doc["question_id"],
            "dataset": doc["dataset"],
            "category": doc["category"],
            "question": doc["question"],
            "answer": doc["gt_answer"],
            "pred": pred,
            "score": eval_results["score"],
            "acc": eval_results["acc"],
            "review": eval_results["review"],
        }
    }
    if pred_dict is not None and "temporal_window" in pred_dict:
        temp_loc_dict = evaluate_temporal_localization(pred_dict["temporal_window"], doc["gt_time_interval"])
        result_dict["gpt_eval_score_acc"]["temp_loc"] = temp_loc_dict
    else:
        raise ValueError(f"No temporal_window found in prediction for question_id {doc['question_id']}. The output must be a dictionary with 'response' and 'temporal_window' keys.")

    return result_dict


def FALCONbench_aggregate_results_mcq(results):
    complete_evaluation_result = {"dataset": {}, "dataset_category": {}, "category": {}}
    to_eval_samples = {"dataset": defaultdict(list), "dataset_category": defaultdict(list), "category": defaultdict(list)}

    temp_metrics = "temp_loc" in results[0]

    for result in results:
        to_eval_samples["dataset"][result["dataset"]].append(result)
        to_eval_samples["dataset_category"][result["category"]].append(result)
        to_eval_samples["category"][result["category"].split("-")[-1]].append(result)

    for split, samples in to_eval_samples.items():
        evaluation_result = {}
        for subset, sub_eval_samples in samples.items():
            judge_dict, metric_dict = evaluate_FALCONbench(sub_eval_samples)
            metric_dict.update({"num_example": len(sub_eval_samples)})
            if temp_metrics:
                metric_dict["temp_loc"] = evaluate_FALCONbench_temporal_localization(sub_eval_samples)
            evaluation_result[subset] = metric_dict
        printable_results = {}

        for cat_name, cat_results in evaluation_result.items():
            printable_results[cat_name] = {
                "num": int(cat_results["num_example"]),
                "acc": round(cat_results["acc"], 5),
            }
        complete_evaluation_result[split] = evaluation_result
        if split == "dataset":
            all_ins_acc, avg_dataset_acc = calculate_ins_level_acc(evaluation_result)
            printable_results["Overall"] = {
                "num": sum([cat_results["num_example"] for cat_results in evaluation_result.values()]),
                "acc": round(all_ins_acc, 5),
                "avg_acc": round(avg_dataset_acc, 5),
            }
            if temp_metrics:
                all_ins_temp = calculate_ins_level_temporal_localization(evaluation_result)
                printable_results["Overall"]["temp_loc"] = all_ins_temp

            eval_logger.info(printable_results)
            complete_evaluation_result["Overall"] = printable_results["Overall"]

    complete_evaluation_result = move_key_first(complete_evaluation_result, "Overall")
    return complete_evaluation_result


def FALCONbench_aggregate_results_oq(results):
    complete_evaluation_result = {"dataset": {}, "dataset_category": {}, "category": {}}
    to_eval_samples = {"dataset": defaultdict(list), "dataset_category": defaultdict(list), "category": defaultdict(list)}

    temp_metrics = "temp_loc" in results[0]

    for result in results:
        to_eval_samples["dataset"][result["dataset"]].append(result)
        to_eval_samples["dataset_category"][result["category"]].append(result)
        to_eval_samples["category"][result["category"].split("-")[-1]].append(result)

    for split, samples in to_eval_samples.items():
        evaluation_result = {}
        for subset, sub_eval_samples in samples.items():
            metric_dict = evaluate_FALCONbench_gpteval(sub_eval_samples)
            metric_dict.update({"num_example": len(sub_eval_samples)})
            if temp_metrics:
                metric_dict["temp_loc"] = evaluate_FALCONbench_temporal_localization(sub_eval_samples)

            evaluation_result[subset] = metric_dict
        printable_results = {}

        for cat_name, cat_results in evaluation_result.items():
            printable_results[cat_name] = {
                "num": int(cat_results["num_example"]),
                "acc": round(cat_results["acc"], 5),
                "score": round(cat_results["score"], 5),
            }

        complete_evaluation_result[split] = evaluation_result
        if split == "dataset":
            all_ins = calculate_ins_level_acc_score(evaluation_result)

            printable_results["Overall"] = {
                "num": sum([cat_results["num_example"] for cat_results in evaluation_result.values()]),
                "acc": round(all_ins["acc"], 5),
                "score": round(all_ins["score"], 5),
                "avg_acc": round(all_ins["avg_acc"], 5),
                "avg_score": round(all_ins["avg_score"], 5),
            }
            if temp_metrics:
                all_ins_temp = calculate_ins_level_temporal_localization(evaluation_result)
                printable_results["Overall"]["temp_loc"] = all_ins_temp

            eval_logger.info(printable_results)
            complete_evaluation_result["Overall"] = printable_results["Overall"]

    complete_evaluation_result = move_key_first(complete_evaluation_result, "Overall")
    return complete_evaluation_result


def FALCONbench_aggregate_results_for_submission(results, args):
    path = generate_submission_file("FALCONbench_test_for_submission.json", args)
    results_dict = {list(item.keys())[0]: list(item.values())[0] for item in results}
    with open(path, "w") as f:
        json.dump(results_dict, f)
    eval_logger.info(f"Results saved to {path}.")
