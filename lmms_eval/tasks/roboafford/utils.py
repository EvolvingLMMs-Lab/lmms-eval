"""RoboAfford-Eval image and mask loading plus official point scoring."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image

ROBOAFFORD_REPO = "tyb197/RoboAfford-Eval"


@lru_cache(maxsize=512)
def _asset_path(relative_path: str) -> str:
    return hf_hub_download(repo_id=ROBOAFFORD_REPO, repo_type="dataset", filename=relative_path)


def roboafford_doc_to_visual(doc: Dict[str, Any]) -> List[Image.Image]:
    """Load the question image as RGB from the official dataset."""
    with Image.open(_asset_path(f"images/{doc['img']}")) as image:
        return [image.convert("RGB")]


def roboafford_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> str:
    """Return the original question, including its coordinate-format instructions."""
    return str(doc["question"])


def _parse_points(answer_text: str) -> List[tuple[float, float]]:
    patterns = [
        r"\((\d+\.?\d*)\s*,\s*(\d+\.?\d*)\)",
        r"\[(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\]",
        r"(\d+\.?\d*)\s+(\d+\.?\d*)",
        r"(\d+\.?\d*),(\d+\.?\d*)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, answer_text)
        if matches:
            return [(float(x), float(y)) for x, y in matches]
    return []


def _point_accuracy(points: List[tuple[float, float]], mask_path: str) -> float:
    if not points:
        return 0.0

    with Image.open(mask_path) as image:
        mask = np.array(image) / 255.0
    if mask.ndim > 2:
        mask = mask[:, :, 0]
    height, width = mask.shape
    normalized = all(0 <= x <= 1 and 0 <= y <= 1 for x, y in points)
    correct = 0
    for x, y in points:
        if normalized:
            x = int(round(x * width))
            y = int(round(y * height))
        else:
            x = int(round(x))
            y = int(round(y))
        if 0 <= x < width and 0 <= y < height and mask[y, x] > 0.5:
            correct += 1
    return correct / len(points)


def roboafford_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Dict[str, Any]]:
    """Score predicted points against the mask and retain the task category."""
    response = results[0] if results else ""
    item = {
        "accuracy": _point_accuracy(_parse_points(response), _asset_path(f"masks/{doc['mask']}")),
        "category": str(doc["category"]),
    }
    return {metric: item for metric in ("roboafford_accuracy", "object_affordance", "object_reference", "spatial_affordance")}


def _mean(results: List[Dict[str, Any]]) -> float:
    return float(np.mean([result["accuracy"] for result in results])) if results else 0.0


def roboafford_aggregate_overall(results: List[Dict[str, Any]]) -> float:
    """Average point accuracy across all questions."""
    return _mean(results)


def _aggregate_category(results: List[Dict[str, Any]], category: str) -> float:
    return _mean([result for result in results if result["category"] == category])


def roboafford_aggregate_object_affordance(results: List[Dict[str, Any]]) -> float:
    """Average object affordance prediction accuracy."""
    return _aggregate_category(results, "object affordance")


def roboafford_aggregate_object_reference(results: List[Dict[str, Any]]) -> float:
    """Average object affordance recognition accuracy."""
    return _aggregate_category(results, "object reference")


def roboafford_aggregate_spatial_affordance(results: List[Dict[str, Any]]) -> float:
    """Average spatial affordance localization accuracy."""
    return _aggregate_category(results, "spatial affordance")
