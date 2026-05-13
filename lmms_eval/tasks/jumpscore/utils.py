import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger as eval_logger


def jumpscore_doc_to_visual(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> List[str]:
    """Return the local video path for a JumpScore sample."""
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    video_ref = str(doc["video_path"])

    if os.path.isabs(video_ref):
        video_path = video_ref
    else:
        hf_home = os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface/"))
        video_cache_dir = lmms_eval_specific_kwargs.get("video_cache_dir", "jumpscore")
        cache_dir = os.path.join(hf_home, video_cache_dir)
        candidates = [
            os.path.join(cache_dir, video_ref),
            os.path.join(cache_dir, "videos", video_ref),
        ]
        video_path = next((path for path in candidates if os.path.exists(path)), candidates[0])

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"JumpScore video path does not exist: {video_path}")

    return [video_path]


def jumpscore_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> str:
    """Build the single-turn JumpScore timestamp prompt."""
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    return f"{pre_prompt}{str(doc['question']).strip()}{post_prompt}"


def jumpscore_doc_to_target(doc: Dict[str, Any]) -> str:
    """Return the raw JumpScore answer string."""
    return str(doc["answer"]).strip()


def jumpscore_doc_to_messages(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Build the multi-turn JumpScore conversation used during evaluation."""
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    video_path = jumpscore_doc_to_visual(doc, lmms_eval_specific_kwargs)[0]
    count_question = str(doc.get("count_question", "")).replace("<image>", "").strip()
    count_question = re.sub(r"\n+", "\n", count_question).strip()
    count_answer = str(doc.get("count_answer", "")).strip()
    timestamps_question = jumpscore_doc_to_text(doc, lmms_eval_specific_kwargs)

    return [
        {
            "role": "user",
            "content": [
                {"type": "video", "url": video_path},
                {"type": "text", "text": count_question},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": count_answer}]},
        {"role": "user", "content": [{"type": "text", "text": timestamps_question}]},
    ]


def is_explicit_empty_gt(gt_data: Any) -> bool:
    """Return whether the raw GT answer is explicitly an empty JSON list."""
    try:
        if isinstance(gt_data, str):
            json_pattern = r"```json\s*(\[.*?\])\s*```|(\[.*?\])"
            json_matches = re.findall(json_pattern, gt_data, re.DOTALL)
            if json_matches:
                json_str = json_matches[0][0] if json_matches[0][0] else json_matches[0][1]
                data = json.loads(json_str)
            else:
                data = json.loads(gt_data)
        else:
            data = gt_data
    except (json.JSONDecodeError, TypeError, AttributeError):
        return False

    return isinstance(data, list) and len(data) == 0


def extract_start_times(paragraph: str) -> List[float]:
    """Extract predicted jump start timestamps from model output."""
    paragraph_lower = paragraph.lower()
    start_times: List[float] = []

    direct_matches = re.findall(r"(?<!\d)(\d+(?:\.\d+)?)\s*s\b", paragraph_lower)
    if direct_matches:
        for ts_str in direct_matches:
            try:
                start_times.append(float(ts_str))
            except ValueError:
                continue
        return sorted(start_times)

    json_pattern = r"```json\s*(\[.*?\])\s*```|(\[.*?\])"
    json_matches = re.findall(json_pattern, paragraph_lower, re.DOTALL)
    if json_matches:
        json_str = None
        for match in json_matches:
            candidate = match[0] if match[0] else match[1]
            if candidate and len(candidate) > len(json_str or ""):
                json_str = candidate

        if json_str:
            json_str = json_str.rstrip()
            json_str = re.sub(r",\s*\]", "]", json_str)
            json_str = re.sub(r",\s*$", "", json_str)
            if not json_str.endswith("]"):
                json_str += "]"

            try:
                data = json.loads(json_str)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, (int, float)):
                            start_times.append(float(item))
                        elif isinstance(item, str):
                            time_str = item.replace("s", "").strip()
                            try:
                                start_times.append(float(time_str))
                            except ValueError:
                                continue
                    if start_times:
                        return sorted(start_times)
            except json.JSONDecodeError:
                bracket_start = json_str.find("[")
                bracket_end = json_str.rfind("]")
                if bracket_start >= 0 and bracket_end > bracket_start:
                    list_content = json_str[bracket_start + 1 : bracket_end]
                    nums = re.findall(r"\b\d+(?:\.\d+)?\b", list_content)
                    start_times = [float(n) for n in nums]
                    if start_times:
                        return sorted(start_times)

    if not start_times:
        nums = re.findall(r"\b\d+(?:\.\d+)?\b", paragraph_lower)
        start_times = [float(n) for n in nums]

    return sorted(start_times)


def parse_gt_start_times(gt_data: Any) -> List[float]:
    """Parse ground-truth JumpScore start timestamps."""
    start_times: List[float] = []
    try:
        if isinstance(gt_data, str):
            json_pattern = r"```json\s*(\[.*?\])\s*```|(\[.*?\])"
            json_matches = re.findall(json_pattern, gt_data, re.DOTALL)
            if json_matches:
                json_str = json_matches[0][0] if json_matches[0][0] else json_matches[0][1]
                data = json.loads(json_str)
            else:
                data = json.loads(gt_data)
        else:
            data = gt_data

        if isinstance(data, list):
            for item in data:
                if isinstance(item, (int, float)):
                    start_times.append(float(item))
                elif isinstance(item, str):
                    time_str = item.replace("s", "").strip()
                    try:
                        start_times.append(float(time_str))
                    except (ValueError, TypeError):
                        continue
    except (json.JSONDecodeError, TypeError, AttributeError) as e:
        eval_logger.warning(f"Failed to parse JumpScore GT start times: {e}")

    return sorted(start_times)


def calculate_map_for_start_times(
    pred_starts: List[float],
    gt_starts: List[float],
    tolerances: List[float],
    confidences: Optional[List[float]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Calculate mAP over start-time predictions under multiple tolerances."""
    if not gt_starts:
        return 0.0, {
            "ap_per_tolerance": {},
            "map": 0.0,
            "num_gt": 0,
            "num_pred": len(pred_starts) if pred_starts else 0,
        }

    if confidences is None:
        confidences = [1.0] * len(pred_starts)

    min_len = min(len(pred_starts), len(confidences))
    pred_starts = pred_starts[:min_len]
    confidences = confidences[:min_len]

    pred_with_conf = list(zip(pred_starts, confidences))
    pred_with_conf.sort(key=lambda x: (-x[1], x[0]))

    ap_per_tolerance: Dict[float, float] = {}
    for tolerance in tolerances:
        tp_count = 0
        fp_count = 0
        matched_gt_indices = set()
        precisions = []
        recalls = []

        for pred_time, _ in pred_with_conf:
            best_match_idx = None
            best_diff = float("inf")

            for i, gt_time in enumerate(gt_starts):
                if i in matched_gt_indices:
                    continue
                diff = abs(pred_time - gt_time)
                if diff <= tolerance and diff < best_diff:
                    best_diff = diff
                    best_match_idx = i

            if best_match_idx is not None:
                tp_count += 1
                matched_gt_indices.add(best_match_idx)
            else:
                fp_count += 1

            precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
            recall = tp_count / len(gt_starts) if gt_starts else 0.0
            precisions.append(precision)
            recalls.append(recall)

        if not recalls:
            ap = 0.0
        else:
            ap = 0.0
            prev_recall = 0.0
            for precision, recall in zip(precisions, recalls):
                ap += precision * (recall - prev_recall)
                prev_recall = recall

        ap_per_tolerance[tolerance] = ap

    map_value = sum(ap_per_tolerance.values()) / len(ap_per_tolerance) if ap_per_tolerance else 0.0
    details = {
        "ap_per_tolerance": ap_per_tolerance,
        "map": map_value,
        "num_gt": len(gt_starts),
        "num_pred": len(pred_starts),
    }
    return map_value, details


def jumpscore_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Dict[str, Any]]:
    """Score one JumpScore prediction with start-time mAP."""
    response = results[0] if len(results) > 0 else ""
    pred_answer_raw = str(response).strip()
    gt_answer_raw = str(doc["answer"]).strip()

    gt_starts = parse_gt_start_times(gt_answer_raw)
    pred_starts = extract_start_times(pred_answer_raw)

    tolerances = [0.1, 0.2, 0.3]
    if not gt_starts and not pred_starts and is_explicit_empty_gt(gt_answer_raw):
        map_value = 1.0
        map_details = {
            "ap_per_tolerance": {tolerance: 1.0 for tolerance in tolerances},
            "map": map_value,
            "num_gt": 0,
            "num_pred": 0,
        }
    else:
        map_value, map_details = calculate_map_for_start_times(
            pred_starts=pred_starts,
            gt_starts=gt_starts,
            tolerances=tolerances,
            confidences=None,
        )

    result = {
        "question_id": doc["id"],
        "map": map_value,
        "ap_per_tolerance": map_details["ap_per_tolerance"],
        "pred_starts": pred_starts,
        "gt_starts": gt_starts,
        "num_pred": map_details["num_pred"],
        "num_gt": map_details["num_gt"],
        "pred_raw": pred_answer_raw[:200] if pred_answer_raw else "",
        "gt_raw": gt_answer_raw[:200] if gt_answer_raw else "",
    }

    return {
        "jumpscore_map": result,
        "jumpscore_score": result.copy(),
    }


def jumpscore_aggregate_results(results: List[Dict[str, Any]]) -> float:
    """Aggregate JumpScore per-sample mAP values."""
    maps = []
    ap_per_tolerance_combined = defaultdict(list)
    bad_pred = 0

    for result in results:
        map_val = float(result.get("map", 0.0))
        maps.append(map_val)
        for tolerance, ap_val in result.get("ap_per_tolerance", {}).items():
            ap_per_tolerance_combined[tolerance].append(float(ap_val))

        if map_val == 0.0 and result.get("pred_starts") == []:
            bad_pred += 1

    if not maps:
        eval_logger.warning("No JumpScore results to aggregate.")
        return 0.0

    mean_map = sum(maps) / len(maps)
    eval_logger.info(f"[JumpScore] Num samples: {len(maps)}\n" f"[JumpScore] Bad pred (no time parsed): {bad_pred}")

    for tolerance in sorted(ap_per_tolerance_combined.keys()):
        ap_list = ap_per_tolerance_combined[tolerance]
        mean_ap = sum(ap_list) / len(ap_list) if ap_list else 0.0
        eval_logger.info(f"[JumpScore] AP@{tolerance}s: {mean_ap:.4f}")

    eval_logger.info(f"[JumpScore] mAP: {mean_map:.4f}")
    return mean_map
