"""PushUpBench: Video repetition counting benchmark.

Evaluates vision-language models on their ability to count exercise repetitions
in videos. The dataset contains workout videos with varying action types, and
models must output the exact count of repetitions.

Dataset fields:
    - name (str): Action description (e.g., "push ups", "leg lift")
    - video_path (str): Video filename (e.g., "xxx_leg_lift_[12].mp4")
    - count (list[int]): Acceptable count values (multiple may be correct)
    - fuzzy_action (bool): Whether the action has ambiguous boundaries
    - complex_action (bool): Whether the action is complex/compound

Metrics:
    - exact_match: Prediction matches any value in ground truth count list
    - mae: Mean Absolute Error between prediction and primary ground truth
    - obo: Off-By-One accuracy (prediction within 1 of any ground truth)
    - r_squared: R² coefficient of determination (outliers |error|>50 excluded)
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from loguru import logger as eval_logger

# Load cache_dir from the YAML config (same pattern as activitynetqa, videomme, etc.)
with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for line in raw_data:
        if "!function" not in line:
            safe_data.append(line)
    config = yaml.safe_load("".join(safe_data))

HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
cache_dir = os.path.join(HF_HOME, config["dataset_kwargs"]["cache_dir"])

NUMBER_WORD_TO_NUMERAL: Dict[str, str] = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
}


def _extract_count(text: Optional[Union[str, int, float]]) -> Optional[int]:
    """Extract a count number from text.

    Handles thinking model outputs by stripping <think>...</think> blocks,
    then tries \\boxed{X} format first, falls back to the last number found.
    """
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None

    # Strip thinking model outputs
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    # Normalize number words
    lowered = text.lower().replace(",", "")
    if lowered in NUMBER_WORD_TO_NUMERAL:
        return int(NUMBER_WORD_TO_NUMERAL[lowered])

    # Try \\boxed{X} format first (preferred structured output)
    boxed_match = re.search(r"\\boxed\{(\d+(?:\.\d+)?)\}", text)
    if boxed_match:
        return int(round(float(boxed_match.group(1))))

    # Fall back to the last number in the text
    all_numbers = re.findall(r"\b(\d+(?:\.\d+)?)\b", text)
    if all_numbers:
        return int(round(float(all_numbers[-1])))

    return None


def _get_gt_counts(doc: Dict[str, Any]) -> List[int]:
    """Get ground truth count list from document.

    Returns list of acceptable integer counts.
    """
    count_field = doc.get("count")
    if count_field is None:
        return []

    # Handle JSON-encoded string (from HF datasets serialization)
    if isinstance(count_field, str):
        try:
            count_field = json.loads(count_field)
        except (json.JSONDecodeError, ValueError):
            parsed = _extract_count(count_field)
            return [parsed] if parsed is not None else []

    if isinstance(count_field, list):
        return [int(c) for c in count_field]

    parsed = _extract_count(count_field)
    return [parsed] if parsed is not None else []


def pushupbench_doc_to_visual(doc: Dict[str, Any]) -> List[str]:
    """Return list containing the video path for this document.

    Videos are automatically downloaded by lmms-eval (via snapshot_download)
    to {HF_HOME}/{cache_dir}/ when video: True is set in dataset_kwargs.
    """
    video_filename = doc["video_path"]
    video_path = os.path.join(cache_dir, video_filename)

    if os.path.exists(video_path):
        return [video_path]

    # Try common extensions
    for ext in ["mp4", "MP4", "mkv", "webm"]:
        alt_path = os.path.splitext(video_path)[0] + "." + ext
        if os.path.exists(alt_path):
            return [alt_path]

    sys.exit(f"Video path: {video_path} does not exist, please check")


def pushupbench_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> str:
    """Format the counting prompt for this document."""
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")

    action_name = str(doc.get("name", "the exercise")).replace("_", " ").strip()
    question = f'Watch this video carefully and count the number of repetitions of "{action_name}". ' f"Provide the count as a single integer."
    return f"{pre_prompt}{question}{post_prompt}"


def pushupbench_doc_to_target(doc: Dict[str, Any]) -> str:
    """Return primary ground truth count as string."""
    counts = _get_gt_counts(doc)
    return str(counts[0]) if counts else ""


def pushupbench_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Dict[str, Any]]:
    """Score a single prediction against ground truth.

    Returns dict with keys for each metric. Each value is a dict containing
    the score and metadata, following lmms-eval conventions.
    """
    pred_text = results[0] if results else ""
    pred_count = _extract_count(pred_text)
    gt_counts = _get_gt_counts(doc)

    if not gt_counts:
        eval_logger.warning(f"No ground truth count for doc: {doc.get('video_path', 'unknown')}")
        entry: Dict[str, Any] = {"score": 0.0, "pred": pred_count, "gt": [], "gt_primary": 0, "mae_value": 0.0}
        return {"exact_match": entry, "mae": entry, "obo": entry, "r_squared": entry}

    gt_primary = gt_counts[0]

    if pred_count is None:
        entry = {"score": 0.0, "pred": None, "gt": gt_counts, "gt_primary": gt_primary}
        return {
            "exact_match": entry,
            "mae": {**entry, "mae_value": float(gt_primary)},
            "obo": entry,
            "r_squared": entry,
        }

    exact = 1.0 if pred_count in gt_counts else 0.0
    mae_value = float(abs(pred_count - gt_primary))
    obo = 1.0 if any(abs(pred_count - gt) <= 1 for gt in gt_counts) else 0.0

    base: Dict[str, Any] = {"pred": pred_count, "gt": gt_counts, "gt_primary": gt_primary}
    return {
        "exact_match": {**base, "score": exact},
        "mae": {**base, "score": mae_value, "mae_value": mae_value},
        "obo": {**base, "score": obo},
        "r_squared": {**base, "score": 0.0},
    }


def pushupbench_aggregate_exact_match(results: List[Dict[str, Any]]) -> float:
    """Aggregate exact match accuracy (percentage)."""
    if not results:
        return 0.0
    total = len(results)
    correct = sum(r["score"] for r in results)
    score = 100.0 * correct / total
    eval_logger.info(f"PushUpBench Exact Match: {score:.2f}% ({int(correct)}/{total})")
    return score


def pushupbench_aggregate_mae(results: List[Dict[str, Any]]) -> float:
    """Aggregate Mean Absolute Error."""
    if not results:
        return 0.0
    score = sum(r["mae_value"] for r in results) / len(results)
    eval_logger.info(f"PushUpBench MAE: {score:.2f}")
    return score


def pushupbench_aggregate_obo(results: List[Dict[str, Any]]) -> float:
    """Aggregate Off-By-One accuracy (percentage)."""
    if not results:
        return 0.0
    total = len(results)
    correct = sum(r["score"] for r in results)
    score = 100.0 * correct / total
    eval_logger.info(f"PushUpBench OBO Accuracy: {score:.2f}%")
    return score


def pushupbench_aggregate_r_squared(results: List[Dict[str, Any]]) -> float:
    """Aggregate R² (coefficient of determination).

    R² = 1 - SS_res / SS_tot where:
        SS_res = sum((gt - pred)²)
        SS_tot = sum((gt - mean(gt))²)

    Excludes outliers where |pred - gt| > 50 to prevent extreme
    prediction failures (e.g., hallucinated counts) from dominating
    the metric.
    """
    if not results:
        return 0.0

    # Collect valid (pred, gt) pairs, excluding outliers
    preds: List[float] = []
    gts: List[float] = []
    n_outliers = 0
    for r in results:
        if r.get("pred") is not None and r.get("gt_primary") is not None:
            p = float(r["pred"])
            g = float(r["gt_primary"])
            if abs(p - g) <= 50:
                preds.append(p)
                gts.append(g)
            else:
                n_outliers += 1

    if len(gts) < 2:
        return 0.0

    mean_gt = sum(gts) / len(gts)
    ss_tot = sum((g - mean_gt) ** 2 for g in gts)
    if ss_tot == 0:
        return 0.0

    ss_res = sum((g - p) ** 2 for g, p in zip(gts, preds))
    r_squared = 1.0 - (ss_res / ss_tot)
    eval_logger.info(f"PushUpBench R²: {r_squared:.4f} (excluded {n_outliers} outliers with |error|>50)")
    return r_squared
