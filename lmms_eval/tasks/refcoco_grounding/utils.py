"""RefCOCO grounding task utilities for lmms-eval.

Grounding-style evaluation: for each image + list of referring expressions,
ask the model to output the bounding boxes for all matching objects as JSON,
then score against the ground-truth bbox with IoU / ACC@t / Center_ACC.

Prompt template (per user spec):
    Locate every object that matches the description "{ref_sentence}" in the
    image. Report bbox coordinates in JSON format.

When there are multiple referring expressions per image (RefCOCO gives a
list), the descriptions are joined into a comma-separated quoted list.

Ground-truth bbox in the HF dataset is [x, y, w, h] in absolute pixels; we
convert to [x1, y1, x2, y2] absolute pixels for scoring. Predicted bboxes are
also expected in absolute pixels (Qwen-VL, InternVL default behavior).
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger as eval_logger


_GROUNDING_METRICS = ["IoU", "ACC@0.1", "ACC@0.3", "ACC@0.5", "ACC@0.7", "ACC@0.9", "Center_ACC"]


def refcoco_grounding_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def _format_refs(answers: Any) -> str:
    """Format the referring expression(s) as they should appear in the prompt.

    Single ref -> "the dog on the left"
    Multiple   -> "the dog on the left", "black dog"
    """
    if isinstance(answers, str):
        return f'"{answers}"'
    if isinstance(answers, list):
        cleaned = [str(a).strip() for a in answers if str(a).strip()]
        if not cleaned:
            return '""'
        return ", ".join(f'"{a}"' for a in cleaned)
    return f'"{answers}"'


def refcoco_grounding_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    refs = _format_refs(doc["answer"])
    return f"Locate every object that matches the description {refs} in the image. Report bbox coordinates in JSON format."


def _xywh_to_xyxy(bbox: List[float]) -> List[float]:
    x, y, w, h = bbox
    return [float(x), float(y), float(x) + float(w), float(y) + float(h)]


def _extract_bbox_from_dict(obj: Dict) -> Optional[List[float]]:
    """Look for common bbox keys in a dict."""
    for key in ("bbox_2d", "bbox", "box", "coordinates", "coord", "location", "position"):
        if key in obj and isinstance(obj[key], (list, tuple)) and len(obj[key]) == 4:
            try:
                return [float(v) for v in obj[key]]
            except (TypeError, ValueError):
                continue
    return None


def _try_parse_json(text: str) -> Optional[Any]:
    """Try several strategies to pull a JSON structure out of a model reply."""
    text = text.strip()
    if not text:
        return None

    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    candidates = [m.group(1)] if m else []
    candidates.append(text)

    for cand in candidates:
        try:
            return json.loads(cand)
        except Exception:
            pass

    for start_ch, end_ch in (("[", "]"), ("{", "}")):
        start = text.find(start_ch)
        end = text.rfind(end_ch)
        if start != -1 and end > start:
            snippet = text[start : end + 1]
            try:
                return json.loads(snippet)
            except Exception:
                continue
    return None


def _bbox_list_from_json(data: Any) -> List[List[float]]:
    """Flatten a JSON structure into a list of [x1,y1,x2,y2]."""
    boxes: List[List[float]] = []
    if data is None:
        return boxes

    if isinstance(data, list):
        if data and all(isinstance(v, (int, float)) for v in data) and len(data) == 4:
            boxes.append([float(v) for v in data])
        else:
            for item in data:
                boxes.extend(_bbox_list_from_json(item))
    elif isinstance(data, dict):
        box = _extract_bbox_from_dict(data)
        if box is not None:
            boxes.append(box)
        else:
            for v in data.values():
                boxes.extend(_bbox_list_from_json(v))
    return boxes


_BBOX_REGEX = re.compile(
    r"\[?\s*(-?\d+(?:\.\d+)?)\s*[,\s]\s*(-?\d+(?:\.\d+)?)\s*[,\s]\s*"
    r"(-?\d+(?:\.\d+)?)\s*[,\s]\s*(-?\d+(?:\.\d+)?)\s*\]?"
)


def _regex_fallback_boxes(text: str) -> List[List[float]]:
    boxes = []
    for m in _BBOX_REGEX.finditer(text):
        boxes.append([float(m.group(i)) for i in range(1, 5)])
    return boxes


def _parse_pred_boxes(pred_text: str) -> List[List[float]]:
    """Parse all bbox candidates from a model response."""
    if not pred_text:
        return []
    data = _try_parse_json(pred_text)
    boxes = _bbox_list_from_json(data)
    if boxes:
        return boxes
    return _regex_fallback_boxes(pred_text)


def _pick_best_box(pred_boxes: List[List[float]], gt_box_xyxy: List[float]) -> List[float]:
    """Choose the predicted box with highest IoU against gt (grounding metric)."""
    if not pred_boxes:
        return [0.0, 0.0, 0.0, 0.0]
    best_box = pred_boxes[0]
    best_iou = -1.0
    for pb in pred_boxes:
        iou = _compute_iou(pb, gt_box_xyxy)
        if iou > best_iou:
            best_iou = iou
            best_box = pb
    return best_box


def _scale_pred_to_pixel(box: List[float], image_w: int, image_h: int) -> List[float]:
    """Qwen-VL grounding boxes are normalized to [0, 1000]. Scale back to pixels."""
    return [
        box[0] / 1000.0 * image_w,
        box[1] / 1000.0 * image_h,
        box[2] / 1000.0 * image_w,
        box[3] / 1000.0 * image_h,
    ]


def refcoco_grounding_process_result(doc, result):
    pred_text = result[0] if result else ""
    gt_xyxy = _xywh_to_xyxy(doc["bbox"])
    image_w, image_h = doc["image"].size
    raw_boxes = _parse_pred_boxes(pred_text)

    # Qwen-VL family outputs boxes normalized to [0, 1000]; other VLMs may
    # emit absolute pixel coordinates. Try both interpretations and keep the
    # one that yields a higher IoU against the ground-truth.
    candidates: List[List[float]] = []
    for pb in raw_boxes:
        candidates.append(pb)
        candidates.append(_scale_pred_to_pixel(pb, image_w, image_h))

    best_pred = _pick_best_box(candidates, gt_xyxy)

    data_dict = {
        "answer": doc["answer"],
        "pred_boxes": raw_boxes,
        "pred": best_pred,
        "bbox": gt_xyxy,
        "ann_id": doc.get("question_id"),
    }
    return {f"refcoco_grounding_{metric}": data_dict for metric in _GROUNDING_METRICS}


def _compute_iou(box1: List[float], box2: List[float]) -> float:
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    inter = max(0.0, x_right - x_left) * max(0.0, y_bottom - y_top)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def _compute_accuracy(box1: List[float], box2: List[float], threshold: float) -> bool:
    return _compute_iou(box1, box2) >= threshold


def _compute_center_accuracy(gt_box: List[float], pred_box: List[float]) -> bool:
    cx = (pred_box[0] + pred_box[2]) / 2
    cy = (pred_box[1] + pred_box[3]) / 2
    return gt_box[0] <= cx <= gt_box[2] and gt_box[1] <= cy <= gt_box[3]


_SCORERS = {
    "IoU": _compute_iou,
    "ACC@0.1": lambda x, y: _compute_accuracy(x, y, 0.1),
    "ACC@0.3": lambda x, y: _compute_accuracy(x, y, 0.3),
    "ACC@0.5": lambda x, y: _compute_accuracy(x, y, 0.5),
    "ACC@0.7": lambda x, y: _compute_accuracy(x, y, 0.7),
    "ACC@0.9": lambda x, y: _compute_accuracy(x, y, 0.9),
    "Center_ACC": _compute_center_accuracy,
}


def _aggregate(results: List[Dict], metric: str) -> float:
    if not results:
        eval_logger.warning(f"No samples for metric refcoco_grounding_{metric}; returning NaN")
        return float("nan")
    fn = _SCORERS[metric]
    scores = [float(fn(r["bbox"], r["pred"])) for r in results]
    return sum(scores) / len(scores)


def refcoco_grounding_iou(results):
    return _aggregate(results, "IoU")


def refcoco_grounding_acc01(results):
    return _aggregate(results, "ACC@0.1")


def refcoco_grounding_acc03(results):
    return _aggregate(results, "ACC@0.3")


def refcoco_grounding_acc05(results):
    return _aggregate(results, "ACC@0.5")


def refcoco_grounding_acc07(results):
    return _aggregate(results, "ACC@0.7")


def refcoco_grounding_acc09(results):
    return _aggregate(results, "ACC@0.9")


def refcoco_grounding_center_acc(results):
    return _aggregate(results, "Center_ACC")
