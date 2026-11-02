"""ShareRobot-Bench adapters and metrics compatible with FlagEvalMM PR #84."""

import ast
import re
from functools import lru_cache
from pathlib import PurePosixPath
from typing import Any

import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image

REPO = "BAAI/ShareRobot-Bench"


def _kind(doc: dict[str, Any]) -> str:
    return "trajectory" if isinstance(doc["image"], list) else "affordance"


def _conversation(doc: dict[str, Any]) -> tuple[str, str]:
    turns = doc["conversations"][:2] if _kind(doc) == "trajectory" else doc["conversations"]
    values = {turn["from"]: turn["value"] for turn in turns}
    if not values.get("human") or not values.get("gpt"):
        raise ValueError(f"Missing question or ground truth for ShareRobot sample {doc['id']}")
    return values["human"], values["gpt"]


@lru_cache(maxsize=512)
def _asset_path(kind: str, image: str) -> str:
    path = PurePosixPath(image)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Image path must be relative to its ShareRobot task folder")
    return hf_hub_download(repo_id=REPO, repo_type="dataset", filename=f"{kind}/{image}")


def _image_path(doc: dict[str, Any]) -> str:
    image = doc["image"]
    if isinstance(image, list):
        if not image:
            raise ValueError("Trajectory sample has no image")
        image = image[0]
    return _asset_path(_kind(doc), image)


def share_robot_doc_to_visual(doc: dict[str, Any]) -> list[Image.Image]:
    """Load the reference image as RGB, using the first trajectory image."""
    with Image.open(_image_path(doc)) as image:
        return [image.convert("RGB")]


def share_robot_doc_to_target(doc: dict[str, Any]) -> str:
    """Return the reference answer from the task's selected conversation turns."""
    return _conversation(doc)[1]


def share_robot_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs: dict[str, Any] | None = None) -> str:
    """Reproduce the reference evaluator's task-specific prompt instructions."""
    question = _conversation(doc)[0]
    if _kind(doc) == "trajectory":
        original = "Please predict up to 10 key trajectory points to complete the task."
        question = question.replace(original, original + " Your answer should be formatted as a list of tuples, i.e. [[x1, y1], [x2, y2], ...], where each tuple contains the x and y coordinates of a point.")
    else:
        question += "\nOnly return the box in format: [x1, y1, x2, y2] with no other output."
    return question.replace("<image>", "")


def share_robot_doc_to_messages(doc: dict[str, Any], lmms_eval_specific_kwargs: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Build one image-and-text chat message without exposing the target answer."""
    return [{"role": "user", "content": [{"type": "image", "url": share_robot_doc_to_visual(doc)[0]}, {"type": "text", "text": share_robot_doc_to_text(doc)}]}]


def _box(value: Any) -> np.ndarray:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    box = np.asarray(value, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError("Expected a finite [x1, y1, x2, y2] box")
    return box


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection = float(np.prod(np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))))
    union = float(np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - intersection)
    return intersection / union if union > 0 else 0.0


def affordance_process_results(doc: dict[str, Any], results: list[str]) -> dict[str, Any]:
    """Score boxes on the reference valid-answer denominator and expose validity."""
    gt = _box(share_robot_doc_to_target(doc))
    if np.any(gt[2:] <= gt[:2]):
        raise ValueError("Ground-truth box must have positive area")
    try:
        value = _iou(_box(results[0] if results else ""), gt)
    except (ValueError, SyntaxError, TypeError, OverflowError):
        value = None
    return {
        "average_iou": value,
        "average_precision": value,
        "iou_at_0_5": None if value is None else float(value >= 0.5),
        "iou_at_0_75": None if value is None else float(value >= 0.75),
        "iou_at_0_9": None if value is None else float(value >= 0.9),
        "valid_items": int(value is not None),
        "valid_rate": float(value is not None),
    }


def share_robot_valid_mean(values: list[float | None]) -> float:
    """Average valid items only, returning the reference zero for no valid items."""
    valid = [value for value in values if value is not None]
    return float(np.mean(valid)) if valid else 0.0


def _points(value: Any, width: int, height: int) -> np.ndarray:
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")

    def normalize(items: Any, separate_axes: bool = False) -> np.ndarray:
        points = np.asarray(items, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("Expected finite two-dimensional points")
        points = points.copy()
        for point in points:
            if separate_axes:
                if point[0] > 1:
                    point[0] /= width
                if point[1] > 1:
                    point[1] /= height
            elif np.any(point > 1):
                point /= [width, height]
        return points

    try:
        return normalize(ast.literal_eval(value) if isinstance(value, str) else value)
    except (ValueError, SyntaxError, TypeError, OverflowError):
        pass
    if not isinstance(value, str):
        return np.empty((0, 2))
    value = value.replace("–", "-").replace("−", "-")
    match = re.search(r"\[(\s*\[[^\]]+\](?:\s*,\s*\[[^\]]+\])*\s*)\]", value)
    if match:
        try:
            return normalize(ast.literal_eval(match.group(0)))
        except (ValueError, SyntaxError, TypeError, OverflowError):
            pass
    pairs = []
    for line in value.splitlines():
        match = re.search(r"\[([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\]", re.sub(r"#.*$", "", line).strip())
        if match:
            pairs.append([float(match[1]), float(match[2])])
    try:
        return normalize(pairs, separate_axes=True) if pairs else np.empty((0, 2))
    except (ValueError, TypeError, OverflowError):
        return np.empty((0, 2))


def _distances(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    distances = np.linalg.norm(pred[:, None, :] - gt[None, :, :], axis=2)
    table = np.empty_like(distances)
    for i in range(len(pred)):
        for j in range(len(gt)):
            if i == j == 0:
                previous = 0.0
            elif i == 0:
                previous = table[i, j - 1]
            elif j == 0:
                previous = table[i - 1, j]
            else:
                previous = min(table[i - 1, j], table[i - 1, j - 1], table[i, j - 1])
            table[i, j] = max(previous, distances[i, j])
    length = min(len(pred), len(gt))
    return float(table[-1, -1]), float(max(distances.min(axis=0).max(), distances.min(axis=1).max())), float(np.sqrt(np.mean(np.sum((pred[:length] - gt[:length]) ** 2, axis=1))))


def trajectory_process_results(doc: dict[str, Any], results: list[str]) -> dict[str, float | int]:
    """Score normalized trajectories, preserving the reference origin fallback."""
    with Image.open(_image_path(doc)) as image:
        width, height = image.size
    gt = _points(share_robot_doc_to_target(doc), width, height)
    if not len(gt):
        raise ValueError(f"Invalid trajectory ground truth for sample {doc['id']}")
    pred = _points(results[0] if results else "", width, height)
    valid = bool(len(pred))
    if not valid:
        pred = np.array([[0.0, 0.0]])
    dfd, hd, rmse = _distances(pred, gt)
    return {"average_discrete_frechet_distance": dfd, "average_hausdorff_distance": hd, "average_root_mean_square_error": rmse, "parse_valid_items": int(valid), "parse_valid_rate": float(valid)}


def share_robot_sum(values: list[int]) -> int:
    """Count valid predictions across all evaluated examples."""
    return sum(values)
