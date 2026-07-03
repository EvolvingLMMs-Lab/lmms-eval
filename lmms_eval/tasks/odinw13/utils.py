import json
import logging
import re
from typing import Any, Dict, List, Tuple

from datasets import Dataset

eval_logger = logging.getLogger("lmms-eval")

PROMPT = (
    "Locate every instance that belongs to the following categories: ´{obj_names}´. "
    "Report bbox coordinates in JSON format."
)

IOU_THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.9]

ODINW13_SUBSETS = [
    "AerialMaritimeDrone",
    "Aquarium",
    "CottontailRabbits",
    "EgoHands",
    "NorthAmericaMushrooms",
    "Packages",
    "PascalVOC",
    "Raccoon",
    "ShellfishOpenImages",
    "VehiclesOpenImages",
    "pistols",
    "pothole",
    "thermalDogsAndPeople",
]


def odinw13_process_docs(dataset: Dataset) -> Dataset:
    """Attach image dimensions so process_results has them without needing the PIL image."""
    return dataset.map(lambda x: {"image_width": x["image"].width, "image_height": x["image"].height})


def odinw13_doc_to_visual(doc: Dict[str, Any]) -> List[Any]:
    return [doc["image"].convert("RGB")]


def odinw13_doc_to_text(doc: Dict[str, Any]) -> str:
    obj_names = ", ".join(doc["categories"])
    return PROMPT.format(obj_names=obj_names)


def _norm_label(name: str) -> str:
    return str(name).strip().lower()


def _extract_all_objects(text: str) -> List[Any]:
    """Extract every top-level {...} object we can parse out of `text`.

    Robust to truncated/malformed JSON: walks the string, tracks brace depth,
    and tries to parse each balanced brace span. Missing outer `]` or missing
    fence closers won't cause us to lose valid inner boxes.
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
    """Pull JSON arrays/objects out of the model's response.

    Preference order: whole response parses; fenced ```json blocks parse;
    else fall back to scraping every parseable {...} object.
    """
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
    """Walk a parsed JSON payload and return a flat list of prediction dicts."""
    results: List[Dict[str, Any]] = []
    stack = [payload]
    while stack:
        node = stack.pop()
        if isinstance(node, list):
            stack.extend(node)
        elif isinstance(node, dict):
            bbox = None
            for key in ("bbox_2d", "bbox", "box_2d", "box"):
                if key in node:
                    bbox = node[key]
                    break
            if isinstance(bbox, list) and len(bbox) == 4:
                label = None
                for key in ("label", "class", "category", "name", "class_name"):
                    if key in node:
                        label = node[key]
                        break
                results.append({"bbox": [float(v) for v in bbox], "label": label})
            else:
                stack.extend(node.values())
    return results


def _parse_predictions(text: str) -> List[Dict[str, Any]]:
    """Parse a model response into a list of {bbox: xyxy, label: str} entries."""
    if not text:
        return []
    parsed: List[Dict[str, Any]] = []
    for block in _extract_json_blocks(text):
        parsed.extend(_flatten_predictions(block))

    if parsed:
        return parsed

    # Fallback: scrape 4-number groups. No labels; downstream matching will treat
    # them as class-agnostic and won't be able to match if labels are enforced.
    quad = re.findall(
        r"(-?\d+(?:\.\d+)?)[\s,]+(-?\d+(?:\.\d+)?)[\s,]+(-?\d+(?:\.\d+)?)[\s,]+(-?\d+(?:\.\d+)?)",
        text,
    )
    return [{"bbox": [float(x) for x in q], "label": None} for q in quad]


def _iou(box_a: List[float], box_b: List[float]) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_w = max(0.0, min(xa2, xb2) - max(xa1, xb1))
    inter_h = max(0.0, min(ya2, yb2) - max(ya1, yb1))
    inter = inter_w * inter_h
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _center_in(gt: List[float], pred: List[float]) -> bool:
    cx = (pred[0] + pred[2]) / 2
    cy = (pred[1] + pred[3]) / 2
    return gt[0] <= cx <= gt[2] and gt[1] <= cy <= gt[3]


def _greedy_match(gts: List[Dict], preds: List[Dict]) -> List[Tuple[int, int, float]]:
    """Greedy 1-to-1 matching within each category.

    Returns list of (gt_idx, pred_idx, iou) for matched pairs. Only pairs with
    the same normalized label are eligible.
    """
    by_label: Dict[str, Tuple[List[int], List[int]]] = {}
    for i, g in enumerate(gts):
        by_label.setdefault(_norm_label(g["label"]), ([], []))[0].append(i)
    for j, p in enumerate(preds):
        lbl = _norm_label(p.get("label"))
        if lbl in by_label:
            by_label[lbl][1].append(j)

    matched: List[Tuple[int, int, float]] = []
    for _label, (gt_ids, pred_ids) in by_label.items():
        if not gt_ids or not pred_ids:
            continue
        pairs = []
        for gi in gt_ids:
            for pj in pred_ids:
                iou = _iou(gts[gi]["bbox"], preds[pj]["bbox"])
                if iou > 0:
                    pairs.append((iou, gi, pj))
        pairs.sort(key=lambda t: t[0], reverse=True)
        used_gt: set = set()
        used_pred: set = set()
        for iou, gi, pj in pairs:
            if gi in used_gt or pj in used_pred:
                continue
            matched.append((gi, pj, iou))
            used_gt.add(gi)
            used_pred.add(pj)
    return matched


def _try_bbox_variants(bbox: List[float], width: int, height: int) -> List[List[float]]:
    """Try several coordinate-space interpretations for a predicted bbox.

    Models emit boxes in different spaces:
      * absolute pixel coords
      * normalized to [0, 1]
      * normalized to [0, 1000] (Qwen-style)
    We try each and pick whichever yields the highest IoU with a same-label GT.
    """
    variants = [list(bbox)]
    hi = max(bbox)
    if hi <= 1.0:
        variants.append([bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height])
    if hi <= 1000.0:
        variants.append([bbox[0] / 1000.0 * width, bbox[1] / 1000.0 * height, bbox[2] / 1000.0 * width, bbox[3] / 1000.0 * height])
    return variants


def _best_variant_pred(pred: Dict, gts: List[Dict], width: int, height: int) -> Dict:
    """Pick the bbox interpretation that yields the highest IoU with any GT
    of the same label. Falls back to the original bbox if no same-label GT.
    """
    lbl = _norm_label(pred.get("label"))
    same_label_gts = [g for g in gts if _norm_label(g["label"]) == lbl]
    if not same_label_gts:
        return {"bbox": pred["bbox"], "label": pred.get("label")}
    best_bbox = pred["bbox"]
    best_iou = -1.0
    for candidate in _try_bbox_variants(pred["bbox"], width, height):
        for g in same_label_gts:
            iou = _iou(g["bbox"], candidate)
            if iou > best_iou:
                best_iou = iou
                best_bbox = candidate
    return {"bbox": best_bbox, "label": pred.get("label")}


def odinw13_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Dict[str, Any]]:
    response = results[0] if results else ""
    raw_preds = _parse_predictions(response)

    width = int(doc["image_width"])
    height = int(doc["image_height"])

    gts = [{"bbox": list(b), "label": lbl} for b, lbl in zip(doc["bboxes"], doc["labels"])]
    preds = [_best_variant_pred(p, gts, width, height) for p in raw_preds]

    matches = _greedy_match(gts, preds)
    matched_by_gt = {gi: (pj, iou) for gi, pj, iou in matches}

    per_gt_iou: List[float] = []
    per_gt_center: List[float] = []
    per_gt_acc = {t: [] for t in IOU_THRESHOLDS}
    for gi, g in enumerate(gts):
        if gi in matched_by_gt:
            pj, iou = matched_by_gt[gi]
            per_gt_iou.append(iou)
            per_gt_center.append(1.0 if _center_in(g["bbox"], preds[pj]["bbox"]) else 0.0)
            for t in IOU_THRESHOLDS:
                per_gt_acc[t].append(1.0 if iou >= t else 0.0)
        else:
            per_gt_iou.append(0.0)
            per_gt_center.append(0.0)
            for t in IOU_THRESHOLDS:
                per_gt_acc[t].append(0.0)

    record = {
        "dataset_name": doc["dataset_name"],
        "n_gt": len(gts),
        "n_pred": len(preds),
        "iou_sum": float(sum(per_gt_iou)),
        "center_sum": float(sum(per_gt_center)),
        "acc_sum": {t: float(sum(v)) for t, v in per_gt_acc.items()},
    }

    metrics = {"odinw13_IoU": record, "odinw13_Center_ACC": record}
    for t in IOU_THRESHOLDS:
        metrics[f"odinw13_ACC@{t}"] = record
    return metrics


def _macro_average_by_dataset(records: List[Dict[str, Any]], key_fn) -> float:
    """Compute per-sub-dataset mean of `key_fn(rec) / rec['n_gt']` and macro-average.

    Only includes sub-datasets with at least one GT box. `key_fn` returns a
    scalar sum for a given per-image record (e.g., record['iou_sum']).
    """
    groups: Dict[str, Tuple[float, int]] = {}
    for rec in records:
        n_gt = rec.get("n_gt", 0)
        if n_gt <= 0:
            continue
        sum_val = key_fn(rec)
        s, n = groups.get(rec["dataset_name"], (0.0, 0))
        groups[rec["dataset_name"]] = (s + sum_val, n + n_gt)

    per_dataset = {name: s / n for name, (s, n) in groups.items() if n > 0}
    for name in ODINW13_SUBSETS:
        eval_logger.info(f"[odinw13] {name}: {per_dataset.get(name, float('nan')):.4f}")
    if not per_dataset:
        return 0.0
    return float(sum(per_dataset.values()) / len(per_dataset))


def odinw13_iou(results: List[Dict[str, Any]]) -> float:
    return _macro_average_by_dataset(results, lambda r: r["iou_sum"])


def odinw13_center_acc(results: List[Dict[str, Any]]) -> float:
    return _macro_average_by_dataset(results, lambda r: r["center_sum"])


def _make_acc_aggregator(threshold: float):
    def _agg(results: List[Dict[str, Any]]) -> float:
        return _macro_average_by_dataset(results, lambda r: r["acc_sum"][threshold])

    _agg.__name__ = f"odinw13_acc_{str(threshold).replace('.', '')}"
    return _agg


odinw13_acc_01 = _make_acc_aggregator(0.1)
odinw13_acc_03 = _make_acc_aggregator(0.3)
odinw13_acc_05 = _make_acc_aggregator(0.5)
odinw13_acc_07 = _make_acc_aggregator(0.7)
odinw13_acc_09 = _make_acc_aggregator(0.9)
