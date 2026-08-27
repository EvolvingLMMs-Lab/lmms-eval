"""SaCo benchmark evaluation utilities.

Implements SAM3-style cgF1 evaluation for text-prompted instance segmentation
with **oracle evaluation** over 3 independent annotators:

* For each sample the model prediction is compared against each annotator's GT
  independently.  The annotator that yields the best mean local-F1 (across 10
  IoU thresholds) is selected — this is the official SAM3 oracle protocol.

* **Classification** — Image-level MCC (is object present?)
* **Localization**  — Positive micro-F1 averaged over 10 IoU thresholds [0.5:0.05:0.95]
* **Combined**      — cgF1 = 100 * pmF1 * IL_MCC

For more details, see:
Paper: https://arxiv.org/abs/2511.16719
Official evaluation: https://github.com/facebookresearch/sam3/tree/main/scripts/eval/gold

All masks are expected in **COCO RLE** format (``{"counts": "...", "size": [H, W]}``).

Dataset schema (``yasserDahou/saco-gold``)::

    id              int                          Global sequential ID
    image           Image                        RGB image
    expression      string                       Referring expression
    human_1_masks   [{size: [H,W], counts: str}] Annotator 1 masks (COCO RLE)
    human_2_masks   [{size: [H,W], counts: str}] Annotator 2 masks (COCO RLE)
    human_3_masks   [{size: [H,W], counts: str}] Annotator 3 masks (COCO RLE)
    count           int                          max(len(h1), len(h2), len(h3))

After running evaluation with lmms-eval, call :func:`compute_saco_final_metrics`
on the results JSON to compute dataset-level MCC, pmF1, and cgF1.
"""

import json
import math
from typing import Any, Dict, List, Tuple

import numpy as np
from loguru import logger as eval_logger
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment

# Official SAM3 IoU thresholds (0.5:0.05:0.95)
IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


# =========================================================================== #
#  Mask helpers (COCO RLE)
# =========================================================================== #


def decode_coco_rle(rle: Dict) -> np.ndarray:
    """Decode a single COCO RLE dict to a binary (H, W) numpy mask."""
    rle_copy = rle.copy()
    if isinstance(rle_copy.get("counts"), str):
        rle_copy["counts"] = rle_copy["counts"].encode("utf-8")
    return mask_utils.decode(rle_copy).astype(np.uint8)


def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """IoU between two binary masks."""
    m1 = (mask1 > 0).astype(np.uint8)
    m2 = (mask2 > 0).astype(np.uint8)
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def mask_nms(masks: List[np.ndarray], iou_threshold: float = 0.9) -> List[np.ndarray]:
    """Remove duplicate masks via greedy NMS (keep larger mask)."""
    if len(masks) <= 1:
        return masks

    areas = [int(np.sum(m > 0)) for m in masks]
    order = np.argsort(areas)[::-1]

    keep, removed = [], set()
    for i in order:
        if i in removed:
            continue
        keep.append(i)
        for j in order:
            if j in removed or j == i or j in keep:
                continue
            if compute_mask_iou(masks[i], masks[j]) >= iou_threshold:
                removed.add(j)

    filtered = [masks[k] for k in sorted(keep)]
    if len(masks) != len(filtered):
        eval_logger.debug(f"Mask NMS: {len(masks)} -> {len(filtered)} " f"(removed {len(masks) - len(filtered)})")
    return filtered


# =========================================================================== #
#  Parsing model output
# =========================================================================== #


def parse_predicted_masks(response_text: str) -> List[Dict]:
    """Extract COCO-RLE mask dicts from the model JSON response.

    Accepts::

        {"masks": [...], "boxes": [...], "scores": [...]}

    or a bare list of mask dicts.
    """
    try:
        data = json.loads(response_text)
        if isinstance(data, dict):
            masks = data.get("masks", [])
        elif isinstance(data, list):
            masks = data
        else:
            return []
        return [m for m in masks if isinstance(m, dict)]
    except (json.JSONDecodeError, TypeError):
        eval_logger.debug("Failed to parse model response as JSON")
        return []


# =========================================================================== #
#  doc_to_* functions (referenced by task YAML)
# =========================================================================== #


def saco_doc_to_visual(doc):
    """Return a list of PIL images for the document."""
    return [doc["image"].convert("RGB")]


def saco_doc_to_text(doc):
    """Return the text prompt (expression)."""
    return doc["expression"]


def saco_doc_to_target(doc):
    """Return ground-truth info needed by process_results.

    Returns all three annotators' masks for oracle evaluation.
    """
    return {
        "human_1_masks": doc.get("human_1_masks", []),
        "human_2_masks": doc.get("human_2_masks", []),
        "human_3_masks": doc.get("human_3_masks", []),
        "count": doc.get("count", 0),
        "expression": doc["expression"],
    }


# =========================================================================== #
#  Per-annotator evaluation helper
# =========================================================================== #


def _evaluate_against_one_gt(
    predicted_masks: List[np.ndarray],
    gt_masks_rle: List[Dict],
) -> Dict[str, Any]:
    """Evaluate predicted (decoded, NMS'd) masks against one annotator's GT.

    Returns a dict with:
        IL_TP/TN/FP/FN  — classification signals
        sample_f1        — mean local F1 across 10 thresholds (-1 for negatives
                           when no GT and no preds → sentinel for macro avg)
        loc_counts       — {"tp": [10], "fp": [10], "fn": [10]}
        has_local_f1s    — True when local_F1s were computed (used by oracle)
        _mean_f1         — mean of local F1 (used by oracle selection)
    """
    target_count = len(gt_masks_rle)
    predicted_count = len(predicted_masks)
    n_thresh = len(IOU_THRESHOLDS)

    is_positive = target_count > 0
    has_prediction = predicted_count > 0

    IL_TP = 1.0 if (is_positive and has_prediction) else 0.0
    IL_TN = 1.0 if (not is_positive and not has_prediction) else 0.0
    IL_FP = 1.0 if (not is_positive and has_prediction) else 0.0
    IL_FN = 1.0 if (is_positive and not has_prediction) else 0.0

    zero_counts = {"tp": [0.0] * n_thresh, "fp": [0.0] * n_thresh, "fn": [0.0] * n_thresh}

    base = {
        "IL_TP": IL_TP,
        "IL_TN": IL_TN,
        "IL_FP": IL_FP,
        "IL_FN": IL_FN,
        "count_accuracy": 1.0 if predicted_count == target_count else 0.0,
    }

    # --- True negative: no GT, no predictions → correct rejection ---------- #
    if not is_positive and not has_prediction:
        base["sample_f1"] = -1.0  # sentinel: excluded from positive macro avg
        base["loc_counts"] = zero_counts
        base["has_local_f1s"] = False
        base["_mean_f1"] = -1.0
        base["_is_true_negative"] = True
        return base

    # --- False positive: no GT but has predictions ------------------------- #
    if not is_positive and has_prediction:
        base["sample_f1"] = -1.0
        base["loc_counts"] = {
            "tp": [0.0] * n_thresh,
            "fp": [0.0] * n_thresh,
            "fn": [0.0] * n_thresh,
        }
        base["has_local_f1s"] = True
        base["_mean_f1"] = 0.0
        base["_is_true_negative"] = False
        return base

    # --- False negative: has GT but no predictions ------------------------- #
    if is_positive and not has_prediction:
        base["sample_f1"] = 0.0
        base["loc_counts"] = {
            "tp": [0.0] * n_thresh,
            "fp": [0.0] * n_thresh,
            "fn": [float(target_count)] * n_thresh,
        }
        base["has_local_f1s"] = True
        base["_mean_f1"] = 0.0
        base["_is_true_negative"] = False
        return base

    # --- True positive: both GT and predictions exist ---------------------- #
    gt_masks = [decode_coco_rle(m) for m in gt_masks_rle]

    n_pred, n_gt = predicted_count, len(gt_masks)
    cost = np.zeros((n_pred, n_gt))
    for i, pm in enumerate(predicted_masks):
        for j, gm in enumerate(gt_masks):
            cost[i, j] = -compute_mask_iou(pm, gm)

    pred_idx, gt_idx = linear_sum_assignment(cost)
    match_ious = -cost[pred_idx, gt_idx]

    tps, fps, fns, local_f1s = [], [], [], []
    for thresh in IOU_THRESHOLDS:
        tp = int((match_ious >= thresh).sum())
        fp = predicted_count - tp
        fn = target_count - tp

        prec = tp / (tp + fp + 1e-4)
        rec = tp / (tp + fn + 1e-4)
        f1 = 2 * prec * rec / (prec + rec + 1e-4)

        tps.append(float(tp))
        fps.append(float(fp))
        fns.append(float(fn))
        local_f1s.append(f1)

    mean_f1 = float(np.mean(local_f1s))
    base["sample_f1"] = mean_f1
    base["loc_counts"] = {"tp": tps, "fp": fps, "fn": fns}
    base["has_local_f1s"] = True
    base["_mean_f1"] = mean_f1
    base["_is_true_negative"] = False
    return base


# =========================================================================== #
#  Oracle selection — pick the best annotator for one sample
# =========================================================================== #


def _select_best_annotator(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return the candidate dict that yields the best oracle score."""
    if len(candidates) == 1:
        return candidates[0]

    best = candidates[0]
    for current in candidates[1:]:
        best_has = best.get("has_local_f1s", False)
        curr_has = current.get("has_local_f1s", False)

        if best_has and curr_has:
            # Both have F1 scores — compare mean F1
            if current["_mean_f1"] > best["_mean_f1"]:
                best = current
        else:
            # One or both lack F1 (true-negative case).
            # Prefer true-negative (IL_TN=1) over other cases.
            if current.get("_is_true_negative", False):
                best = current

    return best


# =========================================================================== #
#  process_results — per-sample metric computation with oracle
#
#  For each sample:
#    1. Parse + decode + NMS the predicted masks (once).
#    2. Evaluate against each of the 3 annotators independently.
#    3. Oracle-select the annotator that yields the best mean local-F1.
#    4. Return that annotator's metrics as the sample result.
# =========================================================================== #


def saco_process_results(doc, result):
    """Compute SAM3-style per-sample metrics with 3-annotator oracle.

    Returns a dict with:

    * ``IL_TP/TN/FP/FN`` — classification signals (from best annotator)
    * ``count_accuracy`` — 1 if predicted count == GT count (best annotator)
    * ``sample_f1`` — average local F1 across 10 IoU thresholds (−1 for negatives)
    * ``loc_counts`` — ``{"tp": [10], "fp": [10], "fn": [10]}`` for post-processing
    """
    response_text = result[0] if len(result) > 0 else ""

    # --- Parse predicted masks (COCO RLE) --------------------------------- #
    predicted_masks_rle = parse_predicted_masks(response_text)

    # Decode predicted masks (shared across all annotator evaluations)
    predicted_masks: List[np.ndarray] = []
    if predicted_masks_rle:
        predicted_masks = [decode_coco_rle(m) for m in predicted_masks_rle]

    predicted_count = len(predicted_masks)
    expression = doc["expression"]
    eval_logger.debug(f"[{expression}] Pred={predicted_count}")

    # --- Evaluate against each annotator ---------------------------------- #
    annotator_keys = ["human_1_masks", "human_2_masks", "human_3_masks"]
    candidates: List[Dict[str, Any]] = []

    for key in annotator_keys:
        gt_masks_rle = doc.get(key, [])
        if gt_masks_rle is None:
            gt_masks_rle = []
        candidate = _evaluate_against_one_gt(predicted_masks, gt_masks_rle)
        candidates.append(candidate)

    # --- Oracle selection ------------------------------------------------- #
    best = _select_best_annotator(candidates)

    # Strip internal oracle fields before returning
    metrics = {
        "IL_TP": best["IL_TP"],
        "IL_TN": best["IL_TN"],
        "IL_FP": best["IL_FP"],
        "IL_FN": best["IL_FN"],
        "count_accuracy": best["count_accuracy"],
        "sample_f1": best["sample_f1"],
        "loc_counts": best["loc_counts"],
    }
    return metrics


# =========================================================================== #
#  Aggregation functions (referenced by task YAML metric_list)
# =========================================================================== #


def IL_TP(items):
    return sum(items) if items else 0.0


def IL_TN(items):
    return sum(items) if items else 0.0


def IL_FP(items):
    return sum(items) if items else 0.0


def IL_FN(items):
    return sum(items) if items else 0.0


def count_accuracy(items):
    return float(np.mean(items)) if items else 0.0


def sample_f1(items):
    """Macro F1: average of per-sample F1 over positive samples only."""
    valid = [x for x in items if x >= 0] if items else []
    return float(np.mean(valid)) if valid else 0.0


def loc_counts(items):
    """Element-wise sum of per-sample TP/FP/FN arrays across 10 thresholds."""
    if not items:
        n = len(IOU_THRESHOLDS)
        return {"tp": [0.0] * n, "fp": [0.0] * n, "fn": [0.0] * n}
    tp = np.zeros(len(IOU_THRESHOLDS))
    fp = np.zeros(len(IOU_THRESHOLDS))
    fn = np.zeros(len(IOU_THRESHOLDS))
    for d in items:
        tp += np.array(d["tp"])
        fp += np.array(d["fp"])
        fn += np.array(d["fn"])
    return {"tp": tp.tolist(), "fp": fp.tolist(), "fn": fn.tolist()}


# =========================================================================== #
#  Derived dataset-level metrics (MCC, pmF1, cgF1)
# =========================================================================== #


def compute_IL_MCC(tp: float, tn: float, fp: float, fn: float) -> float:
    """Matthews Correlation Coefficient."""
    num = tp * tn - fp * fn
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return num / den if den != 0 else 0.0


def compute_pmF1(tp: float, fp: float, fn: float) -> float:
    """Positive micro-F1."""
    den = 2 * tp + fp + fn
    return (2 * tp) / den if den != 0 else 0.0


def compute_cgF1(pmf1: float, mcc: float) -> float:
    """Classification-gated F1 = 100 * pmF1 * IL_MCC."""
    return 100.0 * pmf1 * mcc


def compute_saco_final_metrics(results_path: str, save: bool = True) -> Dict:
    """Compute dataset-level MCC, pmF1, and cgF1 from an lmms-eval results JSON.

    Reads the aggregated classification sums and per-threshold TP/FP/FN
    from the JSON, computes the official SAM3 metrics, optionally writes
    them back, and returns a summary dict.

    Args:
        results_path: Path to the ``*_results.json`` file produced by lmms-eval.
        save: If True, enrich the JSON in-place with the derived metrics.

    Returns:
        ``{task_name: {"IL_MCC": …, "pmF1": …, "cgF1": …, …}}``
    """
    with open(results_path, "r") as f:
        data = json.load(f)

    summary: Dict = {}

    for task_name, task_results in data.get("results", {}).items():
        if not (task_name.startswith("saco_gold") or task_name.startswith("pbench")):
            continue

        # Skip group-level entries that have no actual metrics
        if "IL_TP,none" not in task_results:
            continue

        # --- Classification ------------------------------------------------ #
        il_tp = task_results.get("IL_TP,none", 0.0)
        il_tn = task_results.get("IL_TN,none", 0.0)
        il_fp = task_results.get("IL_FP,none", 0.0)
        il_fn = task_results.get("IL_FN,none", 0.0)

        mcc = compute_IL_MCC(il_tp, il_tn, il_fp, il_fn)

        il_prec = il_tp / (il_tp + il_fp) if (il_tp + il_fp) > 0 else 0.0
        il_rec = il_tp / (il_tp + il_fn) if (il_tp + il_fn) > 0 else 0.0
        il_f1 = 2 * il_prec * il_rec / (il_prec + il_rec) if (il_prec + il_rec) > 0 else 0.0

        # --- Localization (from aggregated loc_counts) --------------------- #
        lc = task_results.get("loc_counts,none", {})
        tp_arr = np.array(lc.get("tp", [0.0] * len(IOU_THRESHOLDS)))
        fp_arr = np.array(lc.get("fp", [0.0] * len(IOU_THRESHOLDS)))
        fn_arr = np.array(lc.get("fn", [0.0] * len(IOU_THRESHOLDS)))

        pmf1_per_thresh = []
        for i in range(len(IOU_THRESHOLDS)):
            pmf1_per_thresh.append(compute_pmF1(tp_arr[i], fp_arr[i], fn_arr[i]))
        pmf1_per_thresh = np.array(pmf1_per_thresh)
        pmf1_avg = float(pmf1_per_thresh.mean())

        # pmF1 at key thresholds
        thresh_idx = {t: i for i, t in enumerate(IOU_THRESHOLDS)}
        pmf1_50 = float(pmf1_per_thresh[thresh_idx[0.5]])
        pmf1_75 = float(pmf1_per_thresh[thresh_idx[0.75]])

        # --- Combined ------------------------------------------------------ #
        cgf1_avg = compute_cgF1(pmf1_avg, mcc)
        cgf1_50 = compute_cgF1(pmf1_50, mcc)
        cgf1_75 = compute_cgF1(pmf1_75, mcc)

        # --- Macro F1 (already aggregated) --------------------------------- #
        macro_f1 = task_results.get("sample_f1,none", 0.0)

        # --- Write back --------------------------------------------------- #
        task_results["IL_MCC,none"] = mcc
        task_results["IL_precision,none"] = il_prec
        task_results["IL_recall,none"] = il_rec
        task_results["IL_F1,none"] = il_f1
        task_results["pmF1,none"] = pmf1_avg
        task_results["pmF1_50,none"] = pmf1_50
        task_results["pmF1_75,none"] = pmf1_75
        task_results["cgF1,none"] = cgf1_avg
        task_results["cgF1_50,none"] = cgf1_50
        task_results["cgF1_75,none"] = cgf1_75

        count_acc = task_results.get("count_accuracy,none", 0.0)

        summary[task_name] = {
            "IL_MCC": mcc,
            "IL_precision": il_prec,
            "IL_recall": il_rec,
            "IL_F1": il_f1,
            "pmF1": pmf1_avg,
            "pmF1_50": pmf1_50,
            "pmF1_75": pmf1_75,
            "cgF1": cgf1_avg,
            "cgF1_50": cgf1_50,
            "cgF1_75": cgf1_75,
            "macro_F1": macro_f1,
            "count_accuracy": count_acc,
        }

    if save:
        with open(results_path, "w") as f:
            json.dump(data, f, indent=2)

    return summary
