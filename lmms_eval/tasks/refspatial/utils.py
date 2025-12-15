import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

PROMPT_SUFFIX_0_999 = (
    "Your answer should be formatted as a list of tuples, i.e. [(x1, y1)], "
    "where each tuple contains the x and y coordinates of a point satisfying the conditions above. "
    "The coordinates should be integers between 0 and 999, representing the pixel locations scaled to a 1000×1000 grid. "
)

FORMAT = "Return only list of tuples, don't add anything else."

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


def refspatial_doc_to_visual(doc: dict) -> list:
    return [doc["image"].convert("RGB")]


# from original repo: https://github.com/Zhoues/RoboRefer/blob/main/Evaluation/summarize_acc.py
def _text2pts(text: str, width: int = 640, height: int = 480, normalization_constant: int = 1, is_absolute: bool = False) -> np.ndarray:
    pattern = r"\(([-+]?\d+\.?\d*(?:,\s*[-+]?\d+\.?\d*)*?)\)"
    matches = re.findall(pattern, text)
    points = []

    for match in matches:
        vector = [float(num) if '.' in num else int(num) for num in match.split(',')]
        if len(vector) == 2:
            x, y = vector
            if not is_absolute:
                x = int(x / normalization_constant * width)
                y = int(y / normalization_constant * height)
            points.append((x, y))

    return np.array(points)


# inspired by original work: https://github.com/Zhoues/RoboRefer/blob/main/Evaluation/summarize_acc.py
def refspatial_process_results(doc: Dict, result: List[str]) -> Dict[str, Dict]:
    key_name = "refspatial_acc"

    mask = np.array(doc["mask"]) / 255.
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

    # process the answer
    acc = 0.0
    if len(points) > 0:
        in_range = (points[:, 0] >= 0) & (points[:, 0] < mask.shape[1]) & \
                    (points[:, 1] >= 0) & (points[:, 1] < mask.shape[0])
        acc = np.concatenate([
            mask[points[in_range, 1], points[in_range, 0]],
            np.zeros(points.shape[0] - in_range.sum())
        ]).mean()

    query = refspatial_doc_to_text(doc)
    omnispatial_submission = {"id": doc["id"], "query": query, "pred": response, "parsed_points": list(map(tuple, points)), "accuracy": acc}
    return {key_name: omnispatial_submission}


def refspatial_aggregate_results(results: List[Dict]) -> float:
    return float(np.mean([sample["accuracy"] for sample in results]))
