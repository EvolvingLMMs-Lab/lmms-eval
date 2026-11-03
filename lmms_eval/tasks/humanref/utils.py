"""HumanRef: multi-instance human referring expression comprehension.

One referring expression can match zero, one or many people. Metrics follow the
official ``metric/recall_precision_densityf1.py`` of ``IDEA-Research/HumanRef``:

* Recall / Precision at IoU 0.5 and averaged over IoU 0.5:0.95 (greedy one-to-one
  matching of ground-truth boxes to predicted boxes),
* DensityF1 = F1 * min(1, gt_count / pred_count), which penalises predicting more
  boxes than there are matching people,
* Rejection score on the ``rejection`` domain: share of expressions for which the
  model predicts no box (the official table reports the absolute count out of 1000).

Box metrics are averaged over the five non-rejection domains (5,000 expressions),
exactly as the official "Average / All Data" row.
"""

import json
import math
import re
import unicodedata
import zipfile
from functools import lru_cache
from io import BytesIO
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from lmms_eval.utils import eval_logger

HUMANREF_REPO = "IDEA-Research/HumanRef"
HUMANREF_REVISION = "9a7b1c1e80ecfa777e2e5af1082120b2f34f31e6"
IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05)
DOMAINS = ("attribute", "position", "interaction", "reasoning", "celebrity")

Box = List[float]

_BOX_KEYS = ("bbox_2d", "bbox", "box", "coordinates", "bounding_box")
_QUAD_RE = re.compile(r"[\[\(]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*[\]\)]")
_EMPTY_ANSWER_RE = re.compile(r"^\s*(?:```(?:json)?\s*)?(?:\[\s*\]|\{\s*\}|null|none)?\s*(?:```)?\s*$", re.IGNORECASE)


# ----------------------------------------------------------------------------
# Images
# ----------------------------------------------------------------------------


def _nfc_basename(name: str) -> str:
    return unicodedata.normalize("NFC", name.rsplit("/", 1)[-1])


@lru_cache(maxsize=1)
def _images_zip_path() -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=HUMANREF_REPO, repo_type="dataset", filename="images.zip", revision=HUMANREF_REVISION)


@lru_cache(maxsize=1)
def _image_member_map() -> Dict[str, str]:
    """Map NFC-normalised basenames to zip members.

    Some celebrity file names (e.g. ``Daniel_Brühl_11.jpg``) are stored without the
    UTF-8 flag, so their cp437-decoded names are recovered as well.
    """
    mapping: Dict[str, str] = {}
    with zipfile.ZipFile(_images_zip_path()) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            base = info.filename.rsplit("/", 1)[-1]
            mapping.setdefault(_nfc_basename(base), info.filename)
            if not info.flag_bits & 0x800:
                try:
                    mapping.setdefault(_nfc_basename(base.encode("cp437").decode("utf-8")), info.filename)
                except (UnicodeEncodeError, UnicodeDecodeError):
                    pass
    return mapping


def _read_image_bytes(image_name: str) -> Optional[bytes]:
    member = _image_member_map().get(_nfc_basename(image_name))
    if member is None:
        eval_logger.warning("humanref: image {} not found in images.zip", image_name)
        return None
    with zipfile.ZipFile(_images_zip_path()) as archive:
        return archive.read(member)


def humanref_doc_to_visual(doc: Dict[str, Any]) -> List[Image.Image]:
    """Load the document image from ``images.zip`` by file name."""
    data = _read_image_bytes(str(doc["image_name"]))
    return [Image.open(BytesIO(data)).convert("RGB")] if data is not None else []


@lru_cache(maxsize=None)
def _image_size(image_name: str) -> Tuple[int, int]:
    """(width, height) from the image header, without decoding pixels."""
    data = _read_image_bytes(image_name)
    if data is None:
        return 1, 1
    with Image.open(BytesIO(data)) as image:
        return image.size


# ----------------------------------------------------------------------------
# Prompt
# ----------------------------------------------------------------------------


def humanref_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> str:
    """Prompt used by the official HumanRef evaluation of Qwen2.5-VL."""
    kwargs = lmms_eval_specific_kwargs or {}
    return f"{kwargs.get('pre_prompt', '')}Locate {doc['referring']}, output its bbox coordinates using JSON format.{kwargs.get('post_prompt', '')}"


# ----------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------


def _final_answer(text: str) -> str:
    return text.rsplit("</think>", 1)[-1].strip()


def _boxes_from_json(data: Any) -> List[Box]:
    boxes: List[Box] = []
    if isinstance(data, dict):
        for key in _BOX_KEYS:
            value = data.get(key)
            if isinstance(value, (list, tuple)) and len(value) == 4 and all(isinstance(v, (int, float)) for v in value):
                return [[float(v) for v in value]]
        for value in data.values():
            boxes.extend(_boxes_from_json(value))
    elif isinstance(data, list):
        if len(data) == 4 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in data):
            return [[float(v) for v in data]]
        for item in data:
            boxes.extend(_boxes_from_json(item))
    return boxes


def _try_json(text: str) -> Optional[Any]:
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(text)
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        start, end = text.find(open_ch), text.rfind(close_ch)
        if start != -1 and end > start:
            candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def parse_boxes(response: str) -> List[Box]:
    """Parse every ``[x1, y1, x2, y2]`` box from a response; ``[]`` means no person.

    JSON output (optionally fenced, with ``bbox_2d`` / ``bbox`` keys or bare lists)
    is preferred; otherwise all bracketed coordinate quadruples are collected.
    """
    text = _final_answer(response)
    if not text or _EMPTY_ANSWER_RE.match(text):
        return []
    boxes = _boxes_from_json(_try_json(text))
    if not boxes:
        boxes = [[float(v) for v in match] for match in _QUAD_RE.findall(text)]
    return [box for box in boxes if box[2] > box[0] and box[3] > box[1]]


def _smart_resize_qwen(height: int, width: int, factor: int = 28, min_pixels: int = 4 * 28 * 28, max_pixels: int = 16384 * 28 * 28) -> Tuple[int, int]:
    """Image size seen by Qwen2.5-VL, whose boxes are absolute in that space."""
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _scale(boxes: Sequence[Box], sx: float, sy: float) -> List[Box]:
    return [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in boxes]


def coordinate_candidates(boxes: Sequence[Box], width: int, height: int) -> List[List[Box]]:
    """Pixel-space readings of a prediction: absolute pixels, a 0-1000 grid, 0-1 and Qwen2.5-VL resized pixels.

    The whole response is read in one convention; the scorer keeps the reading
    with the best DensityF1, so models are not penalised for their native format.
    """
    candidates = [list(boxes), _scale(boxes, width / 1000.0, height / 1000.0)]
    if boxes and max(max(b) for b in boxes) <= 1.0:
        candidates.append(_scale(boxes, width, height))
    resized_h, resized_w = _smart_resize_qwen(height, width)
    candidates.append(_scale(boxes, width / resized_w, height / resized_h))
    return candidates


# ----------------------------------------------------------------------------
# Official metric (ported from metric/recall_precision_densityf1.py)
# ----------------------------------------------------------------------------


def _iou(box1: Box, box2: Box) -> float:
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return intersection / (area1 + area2 - intersection)


def recall_precision(gt_boxes: Sequence[Box], pred_boxes: Sequence[Box], iou_threshold: float) -> Tuple[float, float]:
    """Greedy matching: each GT box takes the best unused prediction with IoU >= threshold."""
    if not pred_boxes:
        return 0.0, 0.0
    matches, used = 0, set()
    for gt_box in gt_boxes:
        best_iou, best_idx = 0.0, -1
        for idx, pred_box in enumerate(pred_boxes):
            if idx in used:
                continue
            iou = _iou(gt_box, pred_box)
            if iou > best_iou and iou >= iou_threshold:
                best_iou, best_idx = iou, idx
        if best_idx != -1:
            matches += 1
            used.add(best_idx)
    return matches / len(gt_boxes), matches / len(pred_boxes)


def density_f1(precision: float, recall: float, gt_count: int, pred_count: int, total_persons: int) -> float:
    """F1 scaled by min(1, gt_count / pred_count); both counts are relative to the persons in the image."""
    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    gt_ratio = gt_count / total_persons if total_persons > 0 else 0.0
    pred_ratio = pred_count / total_persons if total_persons > 0 else 0.0
    return f1 * min(1.0, gt_ratio / pred_ratio) if pred_ratio > 0 else 0.0


def box_metrics(gt_boxes: Sequence[Box], pred_boxes: Sequence[Box], total_persons: int) -> Dict[str, float]:
    """All official box metrics for one non-rejection expression."""
    recalls, precisions, f1s = [], [], []
    for threshold in IOU_THRESHOLDS:
        recall, precision = recall_precision(gt_boxes, pred_boxes, threshold)
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(density_f1(precision, recall, len(gt_boxes), len(pred_boxes), total_persons))
    return {
        "recall_50": recalls[0],
        "precision_50": precisions[0],
        "density_f1_50": f1s[0],
        "recall": float(np.mean(recalls)),
        "precision": float(np.mean(precisions)),
        "density_f1": float(np.mean(f1s)),
    }


# ----------------------------------------------------------------------------
# lmms-eval hooks
# ----------------------------------------------------------------------------

_METRICS = (
    "humanref_recall@0.5",
    "humanref_precision@0.5",
    "humanref_density_f1@0.5",
    "humanref_recall@0.5:0.95",
    "humanref_precision@0.5:0.95",
    "humanref_density_f1@0.5:0.95",
    "humanref_rejection_acc",
    *(f"humanref_{domain}_density_f1@0.5:0.95" for domain in DOMAINS),
)


def score_prediction(doc: Dict[str, Any], pred_boxes: Sequence[Box], width: int, height: int) -> Dict[str, Any]:
    """Per-expression record consumed by every aggregation."""
    record: Dict[str, Any] = {"id": doc["id"], "domain": doc["domain"], "sub_domain": doc["sub_domain"], "num_pred": len(pred_boxes)}
    if doc["domain"] == "rejection":
        record["rejected"] = len(pred_boxes) == 0
        return record
    gt_boxes = [[float(v) for v in box] for box in doc["answer_boxes"]]
    total_persons = len(doc["candidate_boxes"])
    best = None
    for candidate in coordinate_candidates(pred_boxes, width, height):
        metrics = box_metrics(gt_boxes, candidate, total_persons)
        if best is None or metrics["density_f1"] > best["density_f1"]:
            best = metrics
    record.update(best)
    return record


def humanref_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Dict[str, Any]]:
    """Parse the response, read it in the best coordinate convention and score it."""
    width, height = _image_size(str(doc["image_name"]))
    record = score_prediction(doc, parse_boxes(results[0] if results else ""), width, height)
    return {metric: record for metric in _METRICS}


def _mean_over(results: List[Dict[str, Any]], key: str, domain: Optional[str] = None) -> float:
    values = [r[key] for r in results if r["domain"] != "rejection" and (domain is None or r["domain"] == domain)]
    return float(np.mean(values)) if values else float("nan")


def humanref_recall_50(results: List[Dict[str, Any]]) -> float:
    return _mean_over(results, "recall_50")


def humanref_precision_50(results: List[Dict[str, Any]]) -> float:
    return _mean_over(results, "precision_50")


def humanref_density_f1_50(results: List[Dict[str, Any]]) -> float:
    return _mean_over(results, "density_f1_50")


def humanref_recall(results: List[Dict[str, Any]]) -> float:
    return _mean_over(results, "recall")


def humanref_precision(results: List[Dict[str, Any]]) -> float:
    return _mean_over(results, "precision")


def humanref_density_f1(results: List[Dict[str, Any]]) -> float:
    return _mean_over(results, "density_f1")


def humanref_rejection_acc(results: List[Dict[str, Any]]) -> float:
    rejection = [r["rejected"] for r in results if r["domain"] == "rejection"]
    return float(np.mean(rejection)) if rejection else float("nan")


def humanref_attribute_density_f1(results: List[Dict[str, Any]]) -> float:
    return _mean_over(results, "density_f1", "attribute")


def humanref_position_density_f1(results: List[Dict[str, Any]]) -> float:
    return _mean_over(results, "density_f1", "position")


def humanref_interaction_density_f1(results: List[Dict[str, Any]]) -> float:
    return _mean_over(results, "density_f1", "interaction")


def humanref_reasoning_density_f1(results: List[Dict[str, Any]]) -> float:
    return _mean_over(results, "density_f1", "reasoning")


def humanref_celebrity_density_f1(results: List[Dict[str, Any]]) -> float:
    return _mean_over(results, "density_f1", "celebrity")
