"""Video-MME-v2: Multi-Modal Evaluation benchmark for video understanding (v2).

Evaluates VLMs on 800 videos with 3200 8-option MCQ questions (A-H) using
grouped non-linear scoring (relevance + logic groups).

Reference: https://github.com/MME-Benchmarks/Video-MME-v2
"""

import ast
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml
from loguru import logger as eval_logger

hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)

with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for line in raw_data:
        if "!function" not in line:
            safe_data.append(line)
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


# ──────────────────────────────────────────────
# Scoring helpers (from official Video-MME-v2)
# ──────────────────────────────────────────────


def cal_relevance(scores):
    """Quadratic scoring for relevance groups."""
    score_map = {0: 0.0, 1: 100.0 / 16, 2: 100.0 * 4 / 16, 3: 100.0 * 9 / 16, 4: 100.0}
    correct_count = sum(scores)
    return score_map.get(correct_count, 0.0), correct_count * 25.0


def cal_logic(scores, group_structure):
    """Chain-based scoring for logic groups."""
    group_structure_list = ast.literal_eval(group_structure)

    last_correct_idx = -1
    for idx, val in enumerate(scores):
        if val:
            last_correct_idx = idx
        else:
            break

    if group_structure_list == [1, 2, 3, 4]:
        score_map = {0: 0.0, 1: 100.0 / 16, 2: 100.0 * 4 / 16, 3: 100.0 * 9 / 16, 4: 100.0}
    elif group_structure_list == [1, [2, 3], 4]:
        score_map = {0: 0.0, 1: 100.0 / 12, 2: 100.0 * 4 / 12, 3: 100.0 * 7 / 12, 4: 100.0}
        if last_correct_idx == 0 and scores[2]:
            last_correct_idx += 1
    elif group_structure_list == [[1, 2], 3, 4]:
        score_map = {0: 0.0, 1: 100.0 / 10, 2: 100.0 * 2 / 10, 3: 100.0 * 5 / 10, 4: 100.0}
        if last_correct_idx == -1 and scores[1]:
            last_correct_idx += 1
    else:
        raise ValueError(f"Unknown group_structure_list: {group_structure_list}")

    return score_map.get(last_correct_idx + 1, 0.0)


# ──────────────────────────────────────────────
# doc_to_visual
# ──────────────────────────────────────────────


def videomme_v2_doc_to_visual(doc):
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_id = doc["video_id"]
    video_path = os.path.join(cache_dir, "data", f"{video_id}.mp4")
    if os.path.exists(video_path):
        return [video_path]
    elif os.path.exists(video_path.replace("mp4", "MP4")):
        return [video_path.replace("mp4", "MP4")]
    elif os.path.exists(video_path.replace("mp4", "mkv")):
        return [video_path.replace("mp4", "mkv")]
    else:
        sys.exit(f"video path: {video_path} does not exist, please check")


# ──────────────────────────────────────────────
# doc_to_text
# ──────────────────────────────────────────────


def videomme_v2_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs and lmms_eval_specific_kwargs.get("format") == "qwen3_vl":
        return _doc_to_text_qwen3vl(doc, lmms_eval_specific_kwargs)

    instruct_prompt = "Select the best answer to the following multiple-choice question based on the video. " "Respond with only the letter (A, B, C, D, E, F, G, or H) of the correct option."
    question = doc["question"]
    options = doc["options"]  # already "A. ...\nB. ...\n...H. ..."

    full_prompt = f"Question: {question}\n{options}\n{instruct_prompt}"
    return full_prompt


def _doc_to_text_qwen3vl(doc, lmms_eval_specific_kwargs=None):
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") if lmms_eval_specific_kwargs else ""
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "") if lmms_eval_specific_kwargs else ""
    question = doc["question"]
    options = doc["options"]
    full_prompt = f"{pre_prompt}{question}\n{options}\n{post_prompt}"
    return full_prompt


# ──────────────────────────────────────────────
# Subtitle support
# ──────────────────────────────────────────────


def load_subtitle_v2(subtitle_path):
    """Load Video-MME-v2 subtitle from a JSONL file.

    Each line is: {"text": "word", "start_time": float, "end_time": float}
    Returns all text concatenated into a single string.
    """
    texts = []
    try:
        with open(subtitle_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    texts.append(entry["text"])
    except FileNotFoundError:
        eval_logger.warning(f"Subtitle file not found: {subtitle_path}")
        return ""
    except Exception as e:
        eval_logger.warning(f"Error loading subtitle {subtitle_path}: {e}")
        return ""
    return " ".join(texts)


def videomme_v2_doc_to_text_subtitle(doc, lmms_eval_specific_kwargs=None):
    """doc_to_text with subtitle prepended (Video-MME-v2 w/ subtitle variant)."""
    if lmms_eval_specific_kwargs and lmms_eval_specific_kwargs.get("format") == "qwen3_vl":
        return _doc_to_text_subtitle_qwen3vl(doc, lmms_eval_specific_kwargs)

    instruct_prompt = "Select the best answer to the following multiple-choice question based on the video. " "Respond with only the letter (A, B, C, D, E, F, G, or H) of the correct option."
    question = doc["question"]
    options = doc["options"]

    # Load subtitle
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_id = doc["video_id"]
    subtitle_path = os.path.join(cache_dir, "subtitle", "subtitle", f"{video_id}.jsonl")
    subtitle_text = load_subtitle_v2(subtitle_path)

    if subtitle_text:
        prefix = f"This video's subtitles are listed below:\n{subtitle_text}\n\n"
    else:
        prefix = ""

    full_prompt = f"{prefix}Question: {question}\n{options}\n{instruct_prompt}"
    return full_prompt


def _doc_to_text_subtitle_qwen3vl(doc, lmms_eval_specific_kwargs=None):
    """Qwen3-VL format with subtitle."""
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") if lmms_eval_specific_kwargs else ""
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "") if lmms_eval_specific_kwargs else ""
    question = doc["question"]
    options = doc["options"]

    # Load subtitle
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_id = doc["video_id"]
    subtitle_path = os.path.join(cache_dir, "subtitle", "subtitle", f"{video_id}.jsonl")
    subtitle_text = load_subtitle_v2(subtitle_path)

    if subtitle_text:
        prefix = f"This video's subtitles are listed below:\n{subtitle_text}\n\n"
    else:
        prefix = ""

    full_prompt = f"{prefix}{pre_prompt}{question}\n{options}\n{post_prompt}"
    return full_prompt


# ──────────────────────────────────────────────
# Answer extraction
# ──────────────────────────────────────────────


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Final Answer:",
        "Best answer:",
        "Best option:",
        "Answer:",
        "Option:",
    ]
    for prefix in answer_prefixes:
        s = s.replace(prefix, "")

    if len(s.split()) > 10 and not re.search("[A-H]", s):
        return ""

    matches = re.search(r"[A-H]", s)
    if matches is None:
        return ""
    return matches[0]


# ──────────────────────────────────────────────
# process_results
# ──────────────────────────────────────────────


def videomme_v2_process_results(doc, results):
    pred = results[0]
    pred_ans = extract_characters_regex(pred)
    gt_ans = doc["answer"]
    score = 1 if pred_ans.upper() == gt_ans.upper() else 0

    data_dict = {
        "video_id": doc["video_id"],
        "question_id": doc["question_id"],
        "group_type": doc["group_type"],
        "group_structure": doc["group_structure"],
        "level": doc.get("level"),
        "second_head": doc.get("second_head"),
        "third_head": doc.get("third_head"),
        "pred_answer": pred_ans,
        "answer": gt_ans,
        "score": score,
    }
    return {"videomme_v2_score": data_dict}


# ──────────────────────────────────────────────
# aggregate_results
# ──────────────────────────────────────────────


def videomme_v2_aggregate_results(results):
    # Group results by video_id
    video_groups = defaultdict(list)
    for r in results:
        video_groups[r["video_id"]].append(r)

    # Collect per-group scores
    all_group_scores = []  # (group_score, naive_score, group_type, level, second_head, third_head)
    for video_id, items in video_groups.items():
        # Sort by question_id suffix (e.g. "001-1" -> 1)
        items.sort(key=lambda x: int(x["question_id"].split("-")[-1]))
        scores = [item["score"] for item in items]
        group_type = items[0]["group_type"]
        group_structure = items[0]["group_structure"]

        # level/second_head/third_head are only on the last question in each group
        level = None
        second_head = None
        third_head = None
        for item in items:
            if item.get("level") is not None:
                level = item["level"]
            if item.get("second_head") is not None:
                second_head = item["second_head"]
            if item.get("third_head") is not None:
                third_head = item["third_head"]

        if group_type == "relevance":
            group_score, naive_score = cal_relevance(scores)
        elif group_type == "logic":
            group_score = cal_logic(scores, group_structure)
            naive_score = sum(scores) * 25.0
        else:
            eval_logger.warning(f"Unknown group_type '{group_type}' for video {video_id}, using naive scoring")
            group_score = sum(scores) * 25.0
            naive_score = group_score

        all_group_scores.append(
            {
                "video_id": video_id,
                "group_score": group_score,
                "naive_score": naive_score,
                "group_type": group_type,
                "level": level,
                "second_head": second_head,
                "third_head": third_head,
            }
        )

    # ── Overall ──
    total_groups = len(all_group_scores)
    overall_score = sum(g["group_score"] for g in all_group_scores) / total_groups if total_groups > 0 else 0.0
    overall_naive = sum(g["naive_score"] for g in all_group_scores) / total_groups if total_groups > 0 else 0.0
    eval_logger.info(f"Overall Group Score: {overall_score:.2f}% (naive: {overall_naive:.2f}%) [{total_groups} groups]")

    # ── Per group_type ──
    for gt in ["relevance", "logic"]:
        subset = [g for g in all_group_scores if g["group_type"] == gt]
        if subset:
            avg = sum(g["group_score"] for g in subset) / len(subset)
            naive_avg = sum(g["naive_score"] for g in subset) / len(subset)
            eval_logger.info(f"  {gt}: {avg:.2f}% (naive: {naive_avg:.2f}%) [{len(subset)} groups]")

    # ── Per level ──
    level_scores = defaultdict(list)
    for g in all_group_scores:
        if g["level"] is not None:
            level_scores[g["level"]].append(g["group_score"])
    for level in sorted(level_scores.keys()):
        scores_list = level_scores[level]
        avg = sum(scores_list) / len(scores_list)
        eval_logger.info(f"  Level {level}: {avg:.2f}% [{len(scores_list)} groups]")

    # ── Per second_head ──
    sh_scores = defaultdict(list)
    for g in all_group_scores:
        if g["second_head"] is not None:
            sh_scores[g["second_head"]].append(g["group_score"])
    for sh in sorted(sh_scores.keys()):
        scores_list = sh_scores[sh]
        avg = sum(scores_list) / len(scores_list)
        eval_logger.info(f"  Second Head [{sh}]: {avg:.2f}% [{len(scores_list)} groups]")

    # ── Per third_head ──
    th_scores = defaultdict(list)
    for g in all_group_scores:
        if g["third_head"] is not None:
            th_scores[g["third_head"]].append(g["group_score"])
    for th in sorted(th_scores.keys()):
        scores_list = th_scores[th]
        avg = sum(scores_list) / len(scores_list)
        eval_logger.info(f"  Third Head [{th}]: {avg:.2f}% [{len(scores_list)} groups]")

    return overall_score


# ──────────────────────────────────────────────
# Reasoning mode prompt
# ──────────────────────────────────────────────


def videomme_v2_doc_to_text_reasoning(doc, lmms_eval_specific_kwargs=None):
    """Reasoning mode prompt - model must show chain-of-thought before answering."""
    reasoning_prompt = (
        "Please perform a detailed reasoning based on the provided video frames to answer the following "
        "multiple-choice question selecting the best option from A through H and providing your final response "
        "strictly in the format: 'Final Answer: <letter>."
    )
    question = doc["question"]
    options = doc["options"]
    full_prompt = f"Question: {question}\n{options}\n{reasoning_prompt}"
    return full_prompt
