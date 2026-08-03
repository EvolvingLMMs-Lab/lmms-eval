"""MultihopSpatial: multi-hop compositional spatial reasoning benchmark.

Each sample is a multiple-choice question paired with a bounding box that
localizes the answer. Models must both pick the correct choice AND localize it,
which exposes the "lucky guess" problem where MCQ accuracy is high but grounding
is poor.

Metrics (all higher-is-better):
  - mcq_acc:   fraction of samples with the correct choice
  - avg_iou:   mean IoU of the predicted box, over MCQ-correct samples only
  - acc50iou:  fraction of samples that are BOTH MCQ-correct AND have IoU >= 0.5

Coordinate protocol (uniform for every model, for fair comparison):
  - All models get the same prompt, which asks for a normalized (0-1) xyxy box
    as {"bbox_2d": [x1, y1, x2, y2]}.
  - Axis order is taken as xyxy exactly as prompted; no yxyx accommodation.
    A model that emits a different axis order is scored as-is (its IoU suffers).
  - Scale is normalized uniformly: many VLMs emit a 0-1000 scale regardless of
    the prompt, so any box with a coordinate > 1 is divided by 1000. This is a
    lossless units conversion applied identically to every model - it changes no
    model's spatial answer and favors none.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"</?(?:ATT|POS|REL)>")


def _remove_tags(question: str) -> str:
    """Strip the <ATT>/<POS>/<REL> relation markers used during annotation."""
    cleaned = _TAG_RE.sub("", question)
    return re.sub(r"\s+", " ", cleaned).strip()


def _build_prompt(question: str) -> str:
    clean_q = _remove_tags(question)
    return (
        f"{clean_q}\n\n"
        "Please respond in the following format:\n"
        'Answer: (your choice, e.g., "(a) object name")\n'
        'Bounding Box: {"bbox_2d": [x1, y1, x2, y2]}\n\n'
        "Important: Use NORMALIZED coordinates (0.0 to 1.0).\n"
        'Example: {"bbox_2d": [0.25, 0.1, 0.75, 0.8]}'
    )


# ---------------------------------------------------------------------------
# lmms-eval doc hooks
# ---------------------------------------------------------------------------


def multihopspatial_doc_to_visual(doc: Dict[str, Any]) -> List[Any]:
    image = doc["image"]
    return [image.convert("RGB")]


def multihopspatial_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs=None) -> str:
    return _build_prompt(doc["question"])


# ---------------------------------------------------------------------------
# MCQ parsing (verbatim from the reference benchmark for reproducibility)
# ---------------------------------------------------------------------------


def _parse_mcq_answer(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"Answer:\s*(\([a-d]\)\s*[^\n]*)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"Answer:\s*([a-d])\)\s*([^\n]*)", text, re.IGNORECASE)
    if m:
        letter, desc = m.group(1).lower(), m.group(2).strip()
        return f"({letter}) {desc}" if desc else f"({letter})"
    m = re.search(r"Answer:\s*([a-d])\s*$", text, re.IGNORECASE | re.MULTILINE)
    if m:
        return f"({m.group(1).lower()})"
    m = re.search(r"(\([a-d]\)\s*[^\n,\[\]]*)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def _extract_choice_letter(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    m = re.search(r"\(([a-d])\)", text, re.IGNORECASE)
    return m.group(1).lower() if m else None


# ---------------------------------------------------------------------------
# Bounding-box parsing and IoU
# ---------------------------------------------------------------------------


# A single coordinate: an unsigned number like "12", "0.5" or ".5". The pattern
# only matches well-formed floats, so float() below can never raise on it.
_NUM = r"(\d*\.?\d+)"
_BBOX2D_RE = r'"bbox_2d"\s*:\s*\[\s*' + r"\s*,\s*".join([_NUM] * 4) + r"\s*\]"
_BBOX_LINE_RE = r"Bounding Box:\s*\[\s*" + r"\s*,\s*".join([_NUM] * 4) + r"\s*\]"
_BBOX_BARE_RE = r"\[\s*" + r"\s*,\s*".join([_NUM] * 4) + r"\s*\]"


def _parse_bbox(response_text: str) -> Optional[List[float]]:
    """Extract an xyxy box, then normalize its scale uniformly.

    Tries the {"bbox_2d": [...]} hint first, then a "Bounding Box: [...]" line,
    then any bare [a, b, c, d]. Interpreted as xyxy exactly as prompted. If any
    coordinate exceeds 1, the whole box is divided by 1000 (0-1000 -> 0-1).
    Malformed numbers yield None rather than raising.
    """
    groups = None
    m = re.search(_BBOX2D_RE, response_text)
    if m is None:
        m = re.search(_BBOX_LINE_RE, response_text, re.IGNORECASE)
    if m is not None:
        groups = m.groups()
    else:
        found = re.findall(_BBOX_BARE_RE, response_text)
        if found:
            groups = found[0]

    if groups is None:
        return None
    try:
        bbox = [float(v) for v in groups]
    except (TypeError, ValueError):
        return None

    if any(v > 1 for v in bbox):
        bbox = [v / 1000.0 for v in bbox]
    return bbox


def _is_valid_norm_bbox(bbox: Optional[List[float]]) -> bool:
    if bbox is None or len(bbox) != 4:
        return False
    x1, y1, x2, y2 = bbox
    if any(v < 0 or v > 1 for v in bbox):
        return False
    return x2 > x1 and y2 > y1


def _image_dims(doc: Dict[str, Any]) -> Tuple[int, int]:
    res = doc.get("image_resolution")
    if isinstance(res, str) and "x" in res:
        try:
            w, h = res.lower().split("x")
            return int(w), int(h)
        except (ValueError, TypeError):
            pass
    image = doc["image"]
    return image.size  # (width, height)


def _calculate_iou(bbox_gt_xywh: List[float], bbox_pred_norm: List[float], img_w: int, img_h: int) -> Optional[float]:
    """gt is [x, y, w, h] in pixels; pred is [x1, y1, x2, y2] normalized 0-1."""
    if bbox_gt_xywh is None or bbox_pred_norm is None:
        return None
    if len(bbox_gt_xywh) != 4 or len(bbox_pred_norm) != 4:
        return None
    try:
        gx, gy, gw, gh = bbox_gt_xywh
        gt = [gx, gy, gx + gw, gy + gh]
        pred = [
            bbox_pred_norm[0] * img_w,
            bbox_pred_norm[1] * img_h,
            bbox_pred_norm[2] * img_w,
            bbox_pred_norm[3] * img_h,
        ]
        ix1, iy1 = max(gt[0], pred[0]), max(gt[1], pred[1])
        ix2, iy2 = min(gt[2], pred[2]), min(gt[3], pred[3])
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_gt = (gt[2] - gt[0]) * (gt[3] - gt[1])
        area_pred = (pred[2] - pred[0]) * (pred[3] - pred[1])
        union = area_gt + area_pred - inter
        if union <= 0:
            return 0.0
        return round(inter / union, 4)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# process_results + aggregation
# ---------------------------------------------------------------------------


def multihopspatial_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    prediction_text = results[0] if results else ""

    prediction = _parse_mcq_answer(prediction_text)
    pred_letter = _extract_choice_letter(prediction)
    gt_letter = _extract_choice_letter(doc.get("answer"))
    mcq_correct = pred_letter is not None and pred_letter == gt_letter

    pred_bbox = _parse_bbox(prediction_text)
    iou = None
    if _is_valid_norm_bbox(pred_bbox):
        img_w, img_h = _image_dims(doc)
        iou = _calculate_iou(doc.get("bbox"), pred_bbox, img_w, img_h)

    record = {
        "mcq_correct": bool(mcq_correct),
        "iou": iou,
        "hop": doc.get("hop"),
        "view": doc.get("view"),
    }
    return {"mcq_acc": record, "acc50iou": record, "avg_iou": record}


def multihopspatial_aggregate_mcq_acc(results: List[Dict[str, Any]]) -> float:
    if not results:
        return 0.0
    correct = sum(1 for r in results if r["mcq_correct"])
    return round(correct / len(results) * 100, 2)


def multihopspatial_aggregate_acc50iou(results: List[Dict[str, Any]]) -> float:
    if not results:
        return 0.0
    hit = sum(1 for r in results if r["mcq_correct"] and r["iou"] is not None and r["iou"] >= 0.5)
    return round(hit / len(results) * 100, 2)


def multihopspatial_aggregate_avg_iou(results: List[Dict[str, Any]]) -> float:
    ious = [r["iou"] for r in results if r["mcq_correct"] and r["iou"] is not None]
    if not ious:
        return 0.0
    return round(sum(ious) / len(ious) * 100, 2)
