import json
import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from datasets import Dataset

eval_logger = logging.getLogger("lmms-eval")

PROMPT = (
    "Locate the {class_name} in the provided image and output their positions and dimensions "
    "using 3D bounding boxes. The results must be in the JSON format: "
    '["bbox_3d":[x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw],"label":"category"].'
)

MAP_METRIC_KEYS = ["mAP", "mAP25", "mAP50"]


def sunrgbd_process_docs(dataset: Dataset) -> Dataset:
    return dataset.map(lambda x, idx: {"image_id": idx}, with_indices=True)


def sunrgbd_doc_to_visual(doc: Dict[str, Any]) -> List[Any]:
    return [doc["image"].convert("RGB")]


def sunrgbd_doc_to_text(doc: Dict[str, Any]) -> str:
    class_name = ", ".join(doc["categories"])
    return PROMPT.format(class_name=class_name)


def _norm_label(name: str) -> str:
    return str(name).strip().lower()


def _extract_all_objects(text: str) -> List[Any]:
    """Extract every top-level {...} object we can parse from ``text``.

    Robust to truncated / malformed JSON: walks the string, tracks brace depth,
    and tries to parse each balanced brace span.
    """
    out: List[Any] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        in_str = False
        escape = False
        end = -1
        for j in range(i, n):
            ch = text[j]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = j + 1
                    break
        if end == -1:
            break
        try:
            out.append(json.loads(text[i:end]))
            i = end
        except json.JSONDecodeError:
            i += 1
    return out


def _extract_json_blocks(text: str) -> List[Any]:
    try:
        return [json.loads(text)]
    except json.JSONDecodeError:
        pass

    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    for cand in fenced:
        try:
            return [json.loads(cand)]
        except json.JSONDecodeError:
            continue

    objs = _extract_all_objects(text)
    if objs:
        return [objs]
    return []


def _flatten_predictions(payload: Any) -> List[Dict[str, Any]]:
    """Walk parsed JSON and collect prediction dicts with a 9-DoF bbox and label."""
    results: List[Dict[str, Any]] = []
    stack = [payload]
    while stack:
        node = stack.pop()
        if isinstance(node, list):
            stack.extend(node)
        elif isinstance(node, dict):
            bbox = None
            for key in ("bbox_3d", "bbox3d", "box_3d", "bbox"):
                if key in node:
                    bbox = node[key]
                    break
            if isinstance(bbox, list) and len(bbox) >= 7:
                label = None
                for key in ("label", "class", "category", "name", "class_name"):
                    if key in node:
                        label = node[key]
                        break
                # Accept 7-DoF (yaw only) or 9-DoF (roll, pitch, yaw). Normalize to 9-DoF.
                if len(bbox) == 7:
                    b = [float(v) for v in bbox[:6]] + [0.0, 0.0, float(bbox[6])]
                elif len(bbox) >= 9:
                    b = [float(v) for v in bbox[:9]]
                else:
                    stack.extend(node.values())
                    continue
                results.append({"bbox": b, "label": label})
            else:
                stack.extend(node.values())
    return results


def _parse_predictions(text: str) -> List[Dict[str, Any]]:
    if not text:
        return []
    parsed: List[Dict[str, Any]] = []
    for block in _extract_json_blocks(text):
        parsed.extend(_flatten_predictions(block))
    return parsed


def _corners_bev(box: List[float]) -> np.ndarray:
    """Return the 4 (x, z) corners of the box's ground-plane rectangle.

    box: [cx, cy, cz, sx, sy, sz, roll, pitch, yaw]
    """
    cx, _cy, cz, sx, _sy, sz, _r, _p, yaw = box
    hx, hz = sx / 2.0, sz / 2.0
    local = np.array([[-hx, -hz], [hx, -hz], [hx, hz], [-hx, hz]], dtype=np.float64)
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    corners = local @ rot.T
    corners[:, 0] += cx
    corners[:, 1] += cz
    return corners


def _iou_3d(a: List[float], b: List[float]) -> float:
    """Rotated 3D IoU: shapely BEV polygon IoU multiplied by vertical overlap ratio."""
    from shapely.geometry import Polygon

    ax, ay, az, asx, asy, asz = a[0], a[1], a[2], a[3], a[4], a[5]
    bx, by, bz, bsx, bsy, bsz = b[0], b[1], b[2], b[3], b[4], b[5]
    if min(asx, asy, asz, bsx, bsy, bsz) <= 0:
        return 0.0

    try:
        pa = Polygon(_corners_bev(a))
        pb = Polygon(_corners_bev(b))
        if not pa.is_valid or not pb.is_valid:
            pa = pa.buffer(0)
            pb = pb.buffer(0)
        inter_area = pa.intersection(pb).area
        if inter_area <= 0:
            return 0.0
    except Exception:
        return 0.0

    a_ymin, a_ymax = ay - asy / 2.0, ay + asy / 2.0
    b_ymin, b_ymax = by - bsy / 2.0, by + bsy / 2.0
    h_inter = max(0.0, min(a_ymax, b_ymax) - max(a_ymin, b_ymin))
    if h_inter <= 0:
        return 0.0

    vol_inter = inter_area * h_inter
    vol_a = asx * asy * asz
    vol_b = bsx * bsy * bsz
    union = vol_a + vol_b - vol_inter
    if union <= 0:
        return 0.0
    return vol_inter / union


def sunrgbd_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Dict[str, Any]]:
    response = results[0] if results else ""
    raw_preds = _parse_predictions(response)

    gts = [{"bbox": list(b), "label": lbl} for b, lbl in zip(doc["bboxes_3d"], doc["labels"])]
    preds = [{"bbox": p["bbox"], "label": p.get("label")} for p in raw_preds]

    record = {
        "image_id": int(doc["image_id"]),
        "gts": gts,
        "preds": preds,
        "n_gt": len(gts),
        "n_pred": len(preds),
    }

    return {f"sunrgbd_{key}": record for key in MAP_METRIC_KEYS}


def _compute_ap_per_category(
    per_image_gt: Dict[int, List[List[float]]],
    per_image_dt: Dict[int, List[Tuple[List[float], float]]],
    iou_thresh: float,
) -> Optional[float]:
    """VOC-style average precision at a single IoU threshold for one category.

    per_image_gt: image_id -> list of GT boxes
    per_image_dt: image_id -> list of (pred_box, score)

    Returns None if the category has no GT.
    """
    total_gt = sum(len(v) for v in per_image_gt.values())
    if total_gt == 0:
        return None

    detections: List[Tuple[float, int, List[float]]] = []
    for img_id, dets in per_image_dt.items():
        for box, score in dets:
            detections.append((score, img_id, box))
    detections.sort(key=lambda x: x[0], reverse=True)

    matched_gts: Dict[int, np.ndarray] = {img_id: np.zeros(len(gts), dtype=bool) for img_id, gts in per_image_gt.items()}

    tp = np.zeros(len(detections), dtype=np.float64)
    fp = np.zeros(len(detections), dtype=np.float64)

    for k, (_score, img_id, box) in enumerate(detections):
        gts = per_image_gt.get(img_id, [])
        if not gts:
            fp[k] = 1.0
            continue
        matched = matched_gts[img_id]
        best_iou = 0.0
        best_j = -1
        for j, gt in enumerate(gts):
            if matched[j]:
                continue
            iou = _iou_3d(box, gt)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= iou_thresh and best_j >= 0:
            tp[k] = 1.0
            matched[best_j] = True
        else:
            fp[k] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / total_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    # 101-point interpolation (COCO style)
    ap = 0.0
    for t in np.linspace(0.0, 1.0, 101):
        mask = recall >= t
        p = precision[mask].max() if mask.any() else 0.0
        ap += p / 101.0
    return float(ap)


def _map_at_iou(records: List[Dict[str, Any]], iou_thresh: float) -> float:
    per_cat_gt: Dict[str, Dict[int, List[List[float]]]] = {}
    per_cat_dt: Dict[str, Dict[int, List[Tuple[List[float], float]]]] = {}

    for rec in records:
        img_id = rec["image_id"]
        for gt in rec["gts"]:
            key = _norm_label(gt["label"])
            per_cat_gt.setdefault(key, {}).setdefault(img_id, []).append(gt["bbox"])
        for pred in rec["preds"]:
            key = _norm_label(pred.get("label"))
            per_cat_dt.setdefault(key, {}).setdefault(img_id, []).append((pred["bbox"], float(pred.get("score", 1.0))))

    aps: List[float] = []
    for cat, gt_map in per_cat_gt.items():
        dt_map = per_cat_dt.get(cat, {})
        ap = _compute_ap_per_category(gt_map, dt_map, iou_thresh)
        if ap is not None:
            aps.append(ap)
    if not aps:
        return 0.0
    return float(sum(aps) / len(aps))


def sunrgbd_map(results: List[Dict[str, Any]]) -> float:
    """COCO-style mAP at IoU thresholds 0.25:0.05:0.5 for 3D detection.

    SUNRGBD papers typically report mAP@0.25 and mAP@0.5. We report the mean
    across the [.25:.05:.50] range as an aggregated ``mAP`` plus the two
    individual thresholds.
    """
    thresholds = [0.25 + 0.05 * i for i in range(6)]
    vals = [_map_at_iou(results, t) for t in thresholds]
    for t, v in zip(thresholds, vals):
        eval_logger.info(f"[sunrgbd] mAP@{t:.2f}: {v:.4f}")
    return float(sum(vals) / len(vals))


def sunrgbd_map25(results: List[Dict[str, Any]]) -> float:
    return _map_at_iou(results, 0.25)


def sunrgbd_map50(results: List[Dict[str, Any]]) -> float:
    return _map_at_iou(results, 0.5)
