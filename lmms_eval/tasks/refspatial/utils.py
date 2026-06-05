import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

from lmms_eval.tasks._task_utils.point_format import parse_point2d

PROMPT_SUFFIX_0_999 = (
    "Your answer should be formatted as a list of tuples, i.e. [(x1, y1)], "
    "where each tuple contains the x and y coordinates of a point satisfying the conditions above. "
    "The coordinates should be integers between 0 and 999, representing the pixel locations scaled to a 1000×1000 grid. "
)

FORMAT = "Return only list of tuples, don't add anything else."

# JSON ("json") variant: replace the dataset's default tuple-format suffix with a
# Qwen-native single-point JSON instruction on a 0-1000 grid. Any task-specific
# clarification suffix is kept.
_SUFFIX_TO_STRIP = "Your answer should be formatted as a list of tuples, i.e. [(x1, y1)], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points in the image."

_JSON_POST_PROMPT = """Pinpoint the SINGLE best 2D point that satisfies the description. Output exactly one coordinate pair in JSON, normalized to [0, 1000]:
```json
[{"point_2d": [x, y], "label": "target"}]
```
Do not output more than one point. Do not include additional text after the JSON block."""

with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))


def refspatial_doc_to_text(doc: dict[str, Any]) -> str:
    if config.get("metadata", {}).get("prompt_suffix_type", {}) == "0_999":
        print(config)
        return f"{doc['prompt']} {PROMPT_SUFFIX_0_999} {FORMAT}"
    return f"{doc['prompt']} {doc['suffix']} {FORMAT}"


def refspatial_doc_to_text_json(doc: dict[str, Any]) -> str:
    """Qwen-native variant: ask for a single point_2d in JSON on a 0-1000 grid."""
    prompt = doc["prompt"]
    suffix = doc.get("suffix", "")
    if suffix and suffix.strip() != _SUFFIX_TO_STRIP.strip():
        prompt = f"{prompt} {suffix}"
    return f"{prompt}\n{_JSON_POST_PROMPT}"


def refspatial_doc_to_visual(doc: dict) -> list:
    return [doc["image"].convert("RGB")]


# from original repo: https://github.com/Zhoues/RoboRefer/blob/main/Evaluation/summarize_acc.py
def _text2pts(text: str, width: int = 640, height: int = 480, normalization_constant: int = 1, is_absolute: bool = False) -> np.ndarray:
    pattern = r"\(([-+]?\d+\.?\d*(?:,\s*[-+]?\d+\.?\d*)*?)\)"
    matches = re.findall(pattern, text)
    points = []

    for match in matches:
        vector = [float(num) if "." in num else int(num) for num in match.split(",")]
        if len(vector) == 2:
            x, y = vector
            if not is_absolute:
                x = int(x / normalization_constant * width)
                y = int(y / normalization_constant * height)
            points.append((x, y))

    return np.array(points)


def _refspatial_acc(mask: np.ndarray, points: np.ndarray) -> float:
    acc = 0.0
    if len(points) > 0:
        in_range = (points[:, 0] >= 0) & (points[:, 0] < mask.shape[1]) & (points[:, 1] >= 0) & (points[:, 1] < mask.shape[0])
        acc = np.concatenate([mask[points[in_range, 1], points[in_range, 0]], np.zeros(points.shape[0] - in_range.sum())]).mean()
    return acc


# inspired by original work: https://github.com/Zhoues/RoboRefer/blob/main/Evaluation/summarize_acc.py
def refspatial_process_results(doc: Dict, result: List[str]) -> Dict[str, Dict]:
    key_name = "refspatial_acc"

    mask = np.array(doc["mask"]) / 255.0
    # some values might not be exactly 255 or 0
    mask = np.round(mask, 0).astype(int)
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    # extract grounded answer
    response = result[0]

    try:
        prompt_suffix_type = config.get("metadata", {}).get("prompt_suffix_type", {})
    except Exception:
        prompt_suffix_type = {}
    normalization_constant = 1000 if prompt_suffix_type == "0_999" else 1
    points = _text2pts(response, mask.shape[1], mask.shape[0], normalization_constant)

    acc = _refspatial_acc(mask, points)
    query = refspatial_doc_to_text(doc)
    omnispatial_submission = {"id": doc["id"], "query": query, "pred": response, "parsed_points": list(map(tuple, points)), "accuracy": acc}
    return {key_name: omnispatial_submission}


def refspatial_process_results_json(doc: Dict, result: List[str]) -> Dict[str, Dict]:
    """Qwen-native variant: parse JSON point_2d; identical mask scoring."""
    mask = np.array(doc["mask"]) / 255.0
    mask = np.round(mask, 0).astype(int)
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    response = result[0]
    points = parse_point2d(response, mask.shape[1], mask.shape[0])
    acc = _refspatial_acc(mask, points)
    submission = {"id": doc["id"], "query": refspatial_doc_to_text_json(doc), "pred": response, "parsed_points": list(map(tuple, points)), "accuracy": acc}
    return {"refspatial_acc": submission}


def refspatial_aggregate_results(results: List[Dict]) -> float:
    return float(np.mean([sample["accuracy"] for sample in results]))
