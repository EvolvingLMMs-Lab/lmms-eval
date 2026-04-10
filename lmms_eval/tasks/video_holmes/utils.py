"""Video-Holmes: A complex video reasoning benchmark inspired by Sherlock Holmes.

Evaluates VLMs on 270 suspense short films with 1,837 questions (6-choice MCQ, A-F)
across 7 question types: SR, IMC, TCI, TA, MHR, PAR, CTI.

Reference: https://arxiv.org/abs/2505.21374
Dataset: https://huggingface.co/datasets/TencentARC/Video-Holmes
"""

import os
import random
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml
from loguru import logger as eval_logger

DATASET_REPO_ID = "TencentARC/Video-Holmes"

hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)

with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for line in raw_data:
        if "!function" not in line:
            safe_data.append(line)
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


@lru_cache(maxsize=1)
def _get_video_dir():
    """Resolve the video data directory from the HF hub cache."""
    from huggingface_hub import snapshot_download

    snapshot_path = snapshot_download(DATASET_REPO_ID, repo_type="dataset", cache_dir=base_cache_dir)
    video_dir = os.path.join(snapshot_path, "videos_cropped")
    # Extract videos.zip if not already extracted
    if not os.path.isdir(video_dir) or len(os.listdir(video_dir)) == 0:
        import zipfile

        zip_path = os.path.join(snapshot_path, "videos.zip")
        if os.path.exists(zip_path):
            eval_logger.info(f"[video_holmes] Extracting videos from {zip_path}...")
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(snapshot_path)
            eval_logger.info(f"[video_holmes] Extracted {len(os.listdir(video_dir))} videos.")
        else:
            eval_logger.warning(f"[video_holmes] videos.zip not found at {zip_path}")
    return video_dir


QUESTION_TYPES = ["SR", "IMC", "TCI", "TA", "MHR", "PAR", "CTI"]


# ──────────────────────────────────────────────
# doc_to_visual
# ──────────────────────────────────────────────


def video_holmes_doc_to_visual(doc):
    video_dir = _get_video_dir()
    video_id = doc["video ID"]
    for ext in ["mp4", "MP4", "mkv", "webm"]:
        video_path = os.path.join(video_dir, f"{video_id}.{ext}")
        if os.path.exists(video_path):
            return [video_path]
    eval_logger.warning(f"[video_holmes] Video not found: {video_id}. Continuing with text-only fallback.")
    return []


# ──────────────────────────────────────────────
# doc_to_text
# ──────────────────────────────────────────────


def _build_options_str(doc):
    options_dict = doc["Options"]
    option_lines = []
    for key in sorted(options_dict.keys()):
        option_lines.append(f"{key}. {options_dict[key]}")
    return "\n".join(option_lines)


def video_holmes_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs and lmms_eval_specific_kwargs.get("format") == "qwen3_vl":
        return _doc_to_text_qwen3vl(doc, lmms_eval_specific_kwargs)

    question = doc["Question"]
    options = _build_options_str(doc)
    instruct_prompt = "Select the best answer to the following multiple-choice " "question based on the video. Respond with only the letter " "(A, B, C, D, E, or F) of the correct option."
    return f"Question: {question}\n{options}\n{instruct_prompt}"


def _doc_to_text_qwen3vl(doc, lmms_eval_specific_kwargs=None):
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") if lmms_eval_specific_kwargs else ""
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "") if lmms_eval_specific_kwargs else ""
    question = doc["Question"]
    options = _build_options_str(doc)
    return f"{pre_prompt}{question}\n{options}\n{post_prompt}"


def video_holmes_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    """Structured chat messages for chat models (recommended)."""
    prompt = video_holmes_doc_to_text(doc, lmms_eval_specific_kwargs)
    content = []
    for video_path in video_holmes_doc_to_visual(doc):
        content.append({"type": "video", "url": video_path})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def video_holmes_doc_to_text_reasoning(doc, lmms_eval_specific_kwargs=None):
    reasoning_prompt = (
        "Please perform a detailed reasoning based on the provided video frames to answer the following "
        "multiple-choice question selecting the best option from A through F and providing your final response "
        "strictly in the format: 'Final Answer: <letter>'."
    )
    question = doc["Question"]
    options = _build_options_str(doc)
    return f"Question: {question}\n{options}\n{reasoning_prompt}"


# ──────────────────────────────────────────────
# Answer extraction
# ──────────────────────────────────────────────


def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D, E, F.
    Adapted from MMMU: https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L10
    """
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D) (E) (F)
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D E F
            if f"{choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A. B. C. D. E. F.
            if f"{choice}." in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)  # -1 will be ignored anyway
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
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index


# ──────────────────────────────────────────────
# process_results
# ──────────────────────────────────────────────


def video_holmes_process_results(doc, results):
    pred = results[0]

    options_dict = doc["Options"]
    all_choices = sorted(options_dict.keys())
    index2ans = {k: v for k, v in options_dict.items()}

    pred_ans = parse_multi_choice_response(pred, all_choices, index2ans)
    gt_ans = doc["Answer"]
    score = 1 if pred_ans.upper() == gt_ans.upper() else 0

    data_dict = {
        "score": score,
        "question_type": doc.get("Question Type", ""),
    }

    result = {"video_holmes_accuracy": data_dict}
    for qtype in QUESTION_TYPES:
        result[f"video_holmes_{qtype}"] = data_dict
    return result


# ──────────────────────────────────────────────
# Aggregation helpers
# ──────────────────────────────────────────────


def video_holmes_aggregate_accuracy(results):
    total = len(results)
    if total == 0:
        return 0.0
    correct = sum(r["score"] for r in results)
    acc = correct / total * 100
    eval_logger.info(f"Video-Holmes Overall Accuracy: {acc:.2f}% [{total} samples]")
    return acc


def _aggregate_by_question_type(results, question_type):
    subset = [r for r in results if r["question_type"] == question_type]
    if not subset:
        return 0.0
    acc = sum(r["score"] for r in subset) / len(subset) * 100
    eval_logger.info(f"Video-Holmes [{question_type}]: {acc:.2f}% [{len(subset)} samples]")
    return acc


# Per-question-type aggregation functions
def video_holmes_aggregate_SR(results):
    return _aggregate_by_question_type(results, "SR")


def video_holmes_aggregate_IMC(results):
    return _aggregate_by_question_type(results, "IMC")


def video_holmes_aggregate_TCI(results):
    return _aggregate_by_question_type(results, "TCI")


def video_holmes_aggregate_TA(results):
    return _aggregate_by_question_type(results, "TA")


def video_holmes_aggregate_MHR(results):
    return _aggregate_by_question_type(results, "MHR")


def video_holmes_aggregate_PAR(results):
    return _aggregate_by_question_type(results, "PAR")


def video_holmes_aggregate_CTI(results):
    return _aggregate_by_question_type(results, "CTI")
