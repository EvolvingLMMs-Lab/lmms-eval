import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from loguru import logger as eval_logger
from PIL import Image

OPTIONS = ["A", "B", "C", "D", "E"]


def _load_task_config():
    with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
        raw_data = f.readlines()

    safe_data = []
    for line in raw_data:
        if "!function" not in line:
            safe_data.append(line)

    return yaml.safe_load("".join(safe_data))


TASK_CONFIG = _load_task_config()
HF_HOME = os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface/"))
CACHE_DIR = os.path.join(HF_HOME, TASK_CONFIG["dataset_kwargs"]["cache_dir"])
DEFAULT_LMMS_KWARGS = TASK_CONFIG.get("lmms_eval_specific_kwargs", {}).get("default", {})
DEFAULT_MAX_NUM_FRAMES = int(DEFAULT_LMMS_KWARGS.get("max_num_frames", 32))
DEFAULT_FRAME_SAMPLING = str(DEFAULT_LMMS_KWARGS.get("frame_sampling", "uniform"))
DEFAULT_POST_PROMPT = DEFAULT_LMMS_KWARGS.get("post_prompt", "\nAnswer with only the option letter (A, B, C, D, or E).")


def _build_option_map(doc):
    options = {}
    for idx, option_letter in enumerate(OPTIONS):
        option_key = f"answer_choice_{idx}"
        option_text = doc.get(option_key)
        if option_text is None:
            continue
        options[option_letter] = str(option_text).strip()
    return options


def _normalize_text_for_matching(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()


def _with_extension_variants(path):
    root, ext = os.path.splitext(path)
    if ext:
        return [path, root + ext.lower(), root + ext.upper(), root + ".mkv", root + ".MKV"]
    return [path + ".mp4", path + ".MP4", path + ".mkv", path + ".MKV"]


def _candidate_video_paths(video_path):
    normalized = str(video_path).strip().lstrip("./")
    basename = os.path.basename(normalized)
    candidate_rel_paths = [
        normalized,
        basename,
        os.path.join("downloads", basename),
        os.path.join("data", normalized),
        os.path.join("data", "downloads", basename),
        os.path.join("videos", basename),
    ]

    candidates = []
    for relative_path in candidate_rel_paths:
        for candidate in _with_extension_variants(os.path.join(CACHE_DIR, relative_path)):
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


def _resolve_video_path(video_path):
    candidates = _candidate_video_paths(video_path)
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Neptune video not found for '{video_path}'. Tried paths: {candidates[:8]} ...")


def _parse_choice_from_response(response, option_map):
    if response is None:
        return ""

    response_text = str(response).strip()
    if not response_text:
        return ""

    upper_text = response_text.upper()

    exact_match = re.fullmatch(r"[\(\[]?\s*([A-E])\s*[\)\]\.:\-]?\s*", upper_text)
    if exact_match:
        return exact_match.group(1)

    parse_patterns = [
        r"ANSWER\s*(?:IS|:)?\s*[\(\[]?\s*([A-E])\s*[\)\]]?",
        r"OPTION\s*([A-E])",
        r"\(([A-E])\)",
        r"\b([A-E])[\)\.]",
    ]
    for pattern in parse_patterns:
        matches = list(re.finditer(pattern, upper_text))
        if matches:
            return matches[-1].group(1)

    if len(upper_text.split()) <= 3:
        short_match = re.search(r"\b([A-E])\b", upper_text)
        if short_match:
            return short_match.group(1)

    normalized_response = _normalize_text_for_matching(response_text)
    for option_letter, option_text in option_map.items():
        normalized_option = _normalize_text_for_matching(option_text)
        if normalized_option and normalized_option in normalized_response:
            return option_letter

    return ""


def _sample_video_frames(video_path, max_num_frames=32, frame_sampling="uniform"):
    from decord import VideoReader, cpu

    video_reader = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frames = len(video_reader)
    if total_frames == 0:
        return []

    max_num_frames = max(1, int(max_num_frames))
    num_frames = min(max_num_frames, total_frames)

    if frame_sampling != "uniform":
        raise ValueError(f"Unsupported frame_sampling='{frame_sampling}'. Expected 'uniform'.")

    if num_frames == total_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()

    frames = video_reader.get_batch(frame_indices).asnumpy()
    return [Image.fromarray(frame).convert("RGB") for frame in frames]


def neptune_doc_to_visual_video(doc):
    return [_resolve_video_path(doc["video_path"])]


def neptune_doc_to_visual_frames(doc, lmms_eval_specific_kwargs=None):
    lmms_eval_specific_kwargs = lmms_eval_specific_kwargs or {}

    max_num_frames = lmms_eval_specific_kwargs.get("max_num_frames", DEFAULT_MAX_NUM_FRAMES)
    try:
        max_num_frames = int(max_num_frames)
    except (TypeError, ValueError):
        max_num_frames = DEFAULT_MAX_NUM_FRAMES

    frame_sampling = str(lmms_eval_specific_kwargs.get("frame_sampling", DEFAULT_FRAME_SAMPLING))

    video_path = _resolve_video_path(doc["video_path"])
    sampled_frames = _sample_video_frames(video_path, max_num_frames=max_num_frames, frame_sampling=frame_sampling)
    return sampled_frames


def neptune_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    lmms_eval_specific_kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", DEFAULT_POST_PROMPT)

    option_map = _build_option_map(doc)
    options_block = "\n".join([f"{option_letter}. {option_map[option_letter]}" for option_letter in OPTIONS if option_letter in option_map])
    question = str(doc.get("question", "")).strip()

    return f"{pre_prompt}{question}\n{options_block}{post_prompt}"


def neptune_doc_to_target(doc):
    answer_id = doc.get("answer_id")
    if isinstance(answer_id, int) and 0 <= answer_id < len(OPTIONS):
        return OPTIONS[answer_id]
    return ""


def neptune_process_results(doc, results):
    prediction = results[0] if results else ""
    option_map = _build_option_map(doc)

    parsed_prediction = _parse_choice_from_response(prediction, option_map)
    gold_answer = neptune_doc_to_target(doc)

    return {
        "neptune_acc": {
            "key": doc.get("key", ""),
            "question_type": doc.get("question_type", "Unknown"),
            "gold": gold_answer,
            "parsed_pred": parsed_prediction,
            "is_correct": parsed_prediction == gold_answer,
        }
    }


def neptune_aggregate_results(results):
    question_type_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for result in results:
        question_type = result.get("question_type", "Unknown")
        is_correct = bool(result.get("is_correct", False))
        question_type_stats[question_type]["total"] += 1
        question_type_stats[question_type]["correct"] += int(is_correct)

    total = 0
    correct = 0
    printable_results = {}

    for question_type in sorted(question_type_stats):
        total_count = question_type_stats[question_type]["total"]
        correct_count = question_type_stats[question_type]["correct"]
        accuracy = (correct_count / total_count) if total_count else 0.0

        printable_results[question_type] = {
            "num": total_count,
            "acc": round(accuracy, 5),
        }

        total += total_count
        correct += correct_count

    overall_accuracy = (correct / total) if total else 0.0
    printable_results["Overall"] = {
        "num": total,
        "acc": round(overall_accuracy, 5),
    }

    eval_logger.info(printable_results)
    return overall_accuracy
