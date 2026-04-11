"""PerceptionComp: A perception-centric video benchmark.

Evaluates VLMs on 279 videos with 1,114 five-choice MCQ questions (A-E)
across 7 categories and 3 difficulty levels.

Reference: https://arxiv.org/abs/2603.26653
Dataset: https://huggingface.co/datasets/hrinnnn/PerceptionComp

NOTE: Some videos (e.g. Monaco_21m-25m.mp4) cause decord to segfault,
killing the process with no traceback (SIGKILL, exit -9). To avoid this,
force the torchvision backend before running:
    export FORCE_QWENVL_VIDEO_READER=torchvision
This must be set via `export` so child processes (e.g. accelerate launch)
inherit it.
"""

import os
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml
from loguru import logger as eval_logger

DATASET_REPO_ID = "hrinnnn/PerceptionComp"

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
    return os.path.join(snapshot_path, "data")


CATEGORIES = [
    "outdoor tour",
    "shopping",
    "sport",
    "variety show",
    "home tour",
    "game",
    "movie",
]

DIFFICULTY_LEVELS = [1, 2, 3]


# ──────────────────────────────────────────────
# doc_to_visual
# ──────────────────────────────────────────────


def perceptioncomp_doc_to_visual(doc):
    video_dir = _get_video_dir()
    video_id = doc["video_id"]
    for ext in ["mp4", "MP4", "mkv", "webm"]:
        video_path = os.path.join(video_dir, f"{video_id}.{ext}")
        if os.path.exists(video_path):
            return [video_path]
    eval_logger.warning(f"[perceptioncomp] Video not found: {video_id}. Continuing with text-only fallback.")
    return []


# ──────────────────────────────────────────────
# doc_to_text
# ──────────────────────────────────────────────


def _build_options(doc):
    """Build list of (label, text) tuples, skipping empty trailing options."""
    labels = ["A", "B", "C", "D", "E", "F"]
    options = []
    for i, label in enumerate(labels):
        choice = doc.get(f"answer_choice_{i}", "")
        if choice is not None and str(choice).strip():
            options.append((label, str(choice).strip()))
    return options


def _build_options_str(doc):
    return "\n".join(f"{label}. {text}" for label, text in _build_options(doc))


def perceptioncomp_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs and lmms_eval_specific_kwargs.get("format") == "qwen3_vl":
        return _doc_to_text_qwen3vl(doc, lmms_eval_specific_kwargs)

    question = doc["question"]
    options = _build_options_str(doc)
    instruct_prompt = "Select the best answer to the following multiple-choice " "question based on the video. Respond with only the letter " "(A, B, C, D, or E) of the correct option."
    return f"Question: {question}\n{options}\n{instruct_prompt}"


def _doc_to_text_qwen3vl(doc, lmms_eval_specific_kwargs=None):
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") if lmms_eval_specific_kwargs else ""
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "") if lmms_eval_specific_kwargs else ""
    question = doc["question"]
    options = _build_options_str(doc)
    return f"{pre_prompt}{question}\n{options}\n{post_prompt}"


def perceptioncomp_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    """Structured chat messages for chat models (recommended)."""
    prompt = perceptioncomp_doc_to_text(doc, lmms_eval_specific_kwargs)
    content = []
    for video_path in perceptioncomp_doc_to_visual(doc):
        content.append({"type": "video", "url": video_path})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def perceptioncomp_doc_to_text_reasoning(doc, lmms_eval_specific_kwargs=None):
    reasoning_prompt = (
        "Please perform a detailed reasoning based on the provided video frames to answer the following "
        "multiple-choice question selecting the best option from A through E and providing your final response "
        "strictly in the format: 'Final Answer: <letter>'."
    )
    question = doc["question"]
    options = _build_options_str(doc)
    return f"Question: {question}\n{options}\n{reasoning_prompt}"


def perceptioncomp_doc_to_messages_reasoning(doc, lmms_eval_specific_kwargs=None):
    """Structured chat messages for reasoning variant."""
    prompt = perceptioncomp_doc_to_text_reasoning(doc, lmms_eval_specific_kwargs)
    content = []
    for video_path in perceptioncomp_doc_to_visual(doc):
        content.append({"type": "video", "url": video_path})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


# ──────────────────────────────────────────────
# Answer extraction
# ──────────────────────────────────────────────


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
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f"{choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A. B. C. D.
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
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index


# ──────────────────────────────────────────────
# process_results
# ──────────────────────────────────────────────


def perceptioncomp_process_results(doc, results):
    pred = results[0]

    options = _build_options(doc)
    all_choices = [label for label, _ in options]
    index2ans = {label: text for label, text in options}

    pred_ans = parse_multi_choice_response(pred, all_choices, index2ans)
    gt_ans = chr(65 + int(doc["answer_id"]))
    score = 1 if pred_ans.upper() == gt_ans.upper() else 0

    data_dict = {
        "score": score,
        "category": doc.get("category", ""),
        "difficulty": doc.get("difficulty", ""),
    }

    result = {"perceptioncomp_accuracy": data_dict}
    for cat in CATEGORIES:
        key = "perceptioncomp_category_" + cat.replace(" ", "_")
        result[key] = data_dict
    for level in DIFFICULTY_LEVELS:
        result[f"perceptioncomp_difficulty_{level}"] = data_dict
    return result


# ──────────────────────────────────────────────
# Aggregation helpers
# ──────────────────────────────────────────────


def perceptioncomp_aggregate_accuracy(results):
    total = len(results)
    if total == 0:
        return 0.0
    correct = sum(r["score"] for r in results)
    acc = correct / total * 100
    eval_logger.info(f"PerceptionComp Overall Accuracy: {acc:.2f}% [{total} samples]")
    return acc


def _aggregate_by_category(results, category):
    subset = [r for r in results if r["category"] == category]
    if not subset:
        return 0.0
    acc = sum(r["score"] for r in subset) / len(subset) * 100
    eval_logger.info(f"PerceptionComp [{category}]: {acc:.2f}% [{len(subset)} samples]")
    return acc


def _aggregate_by_difficulty(results, difficulty):
    subset = [r for r in results if str(r["difficulty"]) == str(difficulty)]
    if not subset:
        return 0.0
    acc = sum(r["score"] for r in subset) / len(subset) * 100
    eval_logger.info(f"PerceptionComp Difficulty {difficulty}: {acc:.2f}% [{len(subset)} samples]")
    return acc


# Per-category aggregation functions
def perceptioncomp_aggregate_category_outdoor_tour(results):
    return _aggregate_by_category(results, "outdoor tour")


def perceptioncomp_aggregate_category_shopping(results):
    return _aggregate_by_category(results, "shopping")


def perceptioncomp_aggregate_category_sport(results):
    return _aggregate_by_category(results, "sport")


def perceptioncomp_aggregate_category_variety_show(results):
    return _aggregate_by_category(results, "variety show")


def perceptioncomp_aggregate_category_home_tour(results):
    return _aggregate_by_category(results, "home tour")


def perceptioncomp_aggregate_category_game(results):
    return _aggregate_by_category(results, "game")


def perceptioncomp_aggregate_category_movie(results):
    return _aggregate_by_category(results, "movie")


# Per-difficulty aggregation functions
def perceptioncomp_aggregate_difficulty_1(results):
    return _aggregate_by_difficulty(results, 1)


def perceptioncomp_aggregate_difficulty_2(results):
    return _aggregate_by_difficulty(results, 2)


def perceptioncomp_aggregate_difficulty_3(results):
    return _aggregate_by_difficulty(results, 3)
