"""RVOS (Referring Video Object Segmentation) stage-1 evaluation.

Datasets: kcz358/RVOS-{Ref-DAVIS17, MeViS-ValidU, Ref-YT-VOS, ReasonVOS}

For each query, the model receives a video and a referring expression, and is
expected to output an XML string with object point trajectories:

    <tracks coords="t0 id0 x0 y0 id1 x1 y1; t1 id0 x2 y2 ...">label</tracks>

We parse the tracks and score them against GT masks:
  - precision / recall / F1 : Hungarian match pred points -> GT points, count
    a match as correct if the pred point lands inside the matched GT object's
    mask on that frame.
  - HOTA (@ alpha=0.5) : mask point-in-mask similarity aggregated with the
    standard HOTA formulation.

The raw per-query predictions are also written to
``<output_dir>/<task>_submission.json`` in the same schema used by the
upstream tracking pipeline, so the accompanying ``eval_sam2.py`` script can
run stage-2 SAM2 mask propagation on them.
"""

from __future__ import annotations

import ast
import json
import os
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml
from loguru import logger as eval_logger
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

HF_HOME = os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface/"))


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _load_task_cache_dir(task_name: str) -> str:
    """Read the task yaml to find its ``dataset_kwargs.cache_dir`` value."""
    task_yaml = Path(__file__).parent / f"{task_name}.yaml"
    if not task_yaml.exists():
        raise FileNotFoundError(task_yaml)
    with open(task_yaml, "r") as f:
        raw = [ln for ln in f.readlines() if "!function" not in ln]
    cfg = yaml.safe_load("".join(raw))
    return cfg["dataset_kwargs"]["cache_dir"]


def _cache_root_for(doc: Dict[str, Any]) -> str:
    """Resolve <hf_home>/<cache_dir> for the current task."""
    task = _current_task(doc)
    return os.path.join(HF_HOME, _load_task_cache_dir(task))


def _current_task(doc: Dict[str, Any]) -> str:
    """Guess task name from doc id prefix.

    Docs carry ``id`` like ``ref-davis17_track_0`` / ``mevis_track_0`` /
    ``ref-yt-vos_track_0`` / ``reasonvos_track_0``.
    """
    pid = str(doc.get("id", ""))
    if pid.startswith("ref-davis17"):
        return "rvos_ref_davis17"
    if pid.startswith("mevis"):
        return "rvos_mevis_valid_u"
    if pid.startswith("ref-yt-vos"):
        return "rvos_ref_yt_vos"
    if pid.startswith("reasonvos"):
        return "rvos_reasonvos"
    raise ValueError(f"Unknown RVOS id prefix: {pid!r}")


# ---------------------------------------------------------------------------
# Visual / prompt
# ---------------------------------------------------------------------------


def rvos_doc_to_visual(doc):
    root = _cache_root_for(doc)
    video_path = os.path.join(root, "videos", f"{doc['video']}.mp4")
    if not os.path.exists(video_path):
        eval_logger.warning(f"[rvos] Video not found: {video_path}")
        return []
    return [video_path]


def rvos_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    lmms_eval_specific_kwargs = lmms_eval_specific_kwargs or {}
    tmpl = lmms_eval_specific_kwargs.get("prompt_template", "Track the {expression}")
    suffix = lmms_eval_specific_kwargs.get("prompt_suffix", "")
    return tmpl.format(expression=doc["expression"]) + suffix


def rvos_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    prompt = rvos_doc_to_text(doc, lmms_eval_specific_kwargs)
    content = []
    for video_path in rvos_doc_to_visual(doc):
        content.append({"type": "video", "url": video_path})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


# ---------------------------------------------------------------------------
# XML parsing (ported from Feilong607/lmms-eval-ov2-tracking)
# ---------------------------------------------------------------------------


_COORD_REGEX = re.compile(r'coords\s*=\s*[\'"]([^\'"]+)[\'"]')
# Match one segment "time obj_id x y [obj_id x y ...]" separated by any of ;, \t, , : (and leading start).
# Trailing/leading spaces around the delimiter are tolerated.
_FRAME_REGEX = re.compile(r'(?:^|[\t:,;])\s*([0-9]+(?:\.[0-9]+)?)\s+([0-9.\s]+?)(?=[\t:,;]|$)')
_POINTS_REGEX = re.compile(r'([0-9]+)\s+([0-9]{1,4})\s+([0-9]{1,4})')


def parse_xml_tracks(text: str, width: int, height: int, video_fps: float) -> List[Dict[str, Any]]:
    grouped = defaultdict(list)
    for coord_match in _COORD_REGEX.finditer(text):
        for frame_match in _FRAME_REGEX.finditer(coord_match.group(1)):
            timestamp = float(frame_match.group(1))
            point_str = frame_match.group(2)
            for pt_match in _POINTS_REGEX.finditer(point_str):
                ix = pt_match.group(1)
                x = float(pt_match.group(2)) / 1000.0 * width
                y = float(pt_match.group(3)) / 1000.0 * height
                if 0 <= x <= width and 0 <= y <= height:
                    grouped[timestamp].append((ix, x, y))
    out = []
    for ts in sorted(grouped):
        frame = round(ts * video_fps)
        points = {}
        for ix, x, y in grouped[ts]:
            if str(ix) not in points:
                points[str(ix)] = {"point": [x, y]}
        out.append({"time": ts, "frame": frame, "points": points})
    return out


def parse_time_dict_tracks(text: str, width: int, height: int, video_fps: float) -> List[Dict[str, Any]]:
    pattern = r"time\s+(\d+\.?\d*)\s*\n\s*(\{[^}]+\})"
    result = []
    for match in re.finditer(pattern, text, re.MULTILINE):
        seconds = float(match.group(1).strip())
        try:
            obj_points = ast.literal_eval(match.group(2).strip())
        except (ValueError, SyntaxError):
            continue
        frame = round(seconds * video_fps)
        points = {}
        for oid, coords in obj_points.items():
            if len(coords) < 2:
                continue
            x, y = coords[0], coords[1]
            if max(x, y) > 100:
                continue
            px = float(x) / 100.0 * width
            py = float(y) / 100.0 * height
            occ = False
            if len(coords) == 3:
                occ = str(coords[2]).strip().lower() in ("yes", "true", "1")
            points[str(int(oid))] = {"point": [px, py], "occluded": occ}
        if points:
            result.append({"time": seconds, "frame": frame, "points": points})
    return result


def extract_tracks(text: str, width: int, height: int, video_fps: float) -> List[Dict[str, Any]]:
    tracks = parse_xml_tracks(text, width, height, video_fps)
    if tracks:
        return tracks
    return parse_time_dict_tracks(text, width, height, video_fps)


# ---------------------------------------------------------------------------
# GT mask loading (from packed masks.zip -> <cache_root>/masks/<video>/<mid>.json)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4096)
def _load_mask_file(cache_root: str, video: str, mask_id: str) -> Any:
    path = os.path.join(cache_root, "masks", video, f"{mask_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def _ann_to_mask(mask_ann):
    from pycocotools import mask as mask_utils

    if isinstance(mask_ann, np.ndarray):
        return mask_ann
    if isinstance(mask_ann, list):
        h, w = mask_ann[0]["size"]
        rle = mask_utils.merge(mask_utils.frPyObjects(mask_ann, h, w))
    elif isinstance(mask_ann["counts"], list):
        rle = mask_utils.frPyObjects(mask_ann, mask_ann["size"][0], mask_ann["size"][1])
    else:
        rle = mask_ann
    return mask_utils.decode(rle)


def _load_gt_masks(doc: Dict[str, Any]) -> Dict[str, List[Any]]:
    """Return {internal_obj_index_str: [rle_or_none, ...]} aligned with GT tracks."""
    cache_root = _cache_root_for(doc)
    video = doc["video"]
    mask_ids = list(doc["mask_ids"])
    out: Dict[str, List[Any]] = {}
    for idx, mid in enumerate(mask_ids):
        data = _load_mask_file(cache_root, video, mid)
        if data is None:
            out[str(idx)] = []
            continue
        # File is {"<internal_obj_id>": [rle, rle, ..., null, ...]}
        key = next(iter(data.keys()))
        out[str(idx)] = data[key]
    return out


def _load_masks_at_frame(gt_masks: Dict[str, List[Any]], frame_idx: int, height: int, width: int,
                         return_dict: bool = False):
    empty = np.zeros((height, width), dtype=bool)
    masks: List[np.ndarray] = []
    masks_by_id: Dict[str, np.ndarray] = {}
    for mid, mask_list in gt_masks.items():
        binary = empty
        if frame_idx < len(mask_list) and mask_list[frame_idx] is not None:
            binary = _ann_to_mask(mask_list[frame_idx]).astype(bool)
        masks.append(binary)
        masks_by_id[str(mid)] = binary
    if return_dict:
        return masks_by_id
    return masks


def _is_point_in_mask(point, mask) -> bool:
    h, w = mask.shape
    x, y = point
    xi, yi = int(round(x)), int(round(y))
    return 0 <= xi < w and 0 <= yi < h and bool(mask[yi, xi])


# ---------------------------------------------------------------------------
# Metrics (ported)
# ---------------------------------------------------------------------------


def _evaluate_frame(pred_points, gt_points, masks):
    ng = len(gt_points)
    np_ = len(pred_points)
    if ng == 0:
        score = float(np_ == 0)
        return score, score, score
    if np_ == 0:
        return 0.0, 0.0, 0.0
    dist_mat = cdist(np.array(pred_points), np.array(gt_points))
    ri, ci = linear_sum_assignment(dist_mat)
    correct = sum(1 for r, c in zip(ri, ci) if c < len(masks) and _is_point_in_mask(pred_points[r], masks[c]))
    p = correct / np_
    r = correct / len(masks)
    f = 2 * p * r / (p + r + 1e-10) if (p + r) > 0 else 0.0
    return p, r, f


def _evaluate_video_tracks_with_masks(pred_tracks, gt_tracks, gt_masks, height, width):
    pred_by = {e["frame"]: e for e in (pred_tracks or [])}
    gt_by = {e["frame"]: e for e in (gt_tracks or [])}
    all_frames = sorted(gt_by.keys())
    if not all_frames:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    per_frame = []
    for fidx in all_frames:
        pf = pred_by.get(fidx)
        gf = gt_by.get(fidx)
        pp = [tuple(v["point"]) for v in pf["points"].values()] if pf else []
        gp = [tuple(v["point"]) for v in gf["points"].values()] if gf else []
        masks = _load_masks_at_frame(gt_masks, fidx, height, width)
        p, r, f = _evaluate_frame(pp, gp, masks)
        per_frame.append((p, r, f))
    if not per_frame:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    return {
        "precision": float(np.mean([x[0] for x in per_frame])),
        "recall": float(np.mean([x[1] for x in per_frame])),
        "f1": float(np.mean([x[2] for x in per_frame])),
    }


class _HOTAMetric:
    def __init__(self):
        self.alpha_thresholds = np.arange(0.05, 1.0, 0.05)

    def prepare_data(self, pred_tracks, gt_tracks, gt_masks, height, width):
        pred_tracks = pred_tracks or []
        gt_tracks = gt_tracks or []
        pred_by = {e["frame"]: e for e in pred_tracks}
        gt_by = {e["frame"]: e for e in gt_tracks}
        all_gt_ids, all_pred_ids = set(), set()
        for e in gt_tracks:
            all_gt_ids.update(e["points"].keys())
        for e in pred_tracks:
            all_pred_ids.update(e["points"].keys())
        gt_map = {str(oid): i for i, oid in enumerate(sorted(all_gt_ids))}
        pred_map = {str(oid): i for i, oid in enumerate(sorted(all_pred_ids))}
        all_frames = sorted(gt_by.keys())
        data = {
            "num_gt_ids": len(all_gt_ids),
            "num_tracker_ids": len(all_pred_ids),
            "num_timesteps": len(all_frames),
            "gt_ids": [],
            "tracker_ids": [],
            "similarity_scores": [],
            "num_gt_dets": 0,
            "num_tracker_dets": 0,
        }
        for fidx in all_frames:
            pf = pred_by.get(fidx)
            gf = gt_by.get(fidx)
            gt_ids_l, gt_pts = [], []
            if gf:
                for oid, pd in sorted(gf["points"].items()):
                    gt_ids_l.append(gt_map[str(oid)])
                    gt_pts.append(pd["point"])
            pred_ids_l, pred_pts = [], []
            if pf:
                for oid, pd in sorted(pf["points"].items()):
                    pred_ids_l.append(pred_map[str(oid)])
                    pred_pts.append(pd["point"])
            gt_arr = np.array(gt_ids_l, dtype=int)
            pred_arr = np.array(pred_ids_l, dtype=int)
            data["gt_ids"].append(gt_arr)
            data["tracker_ids"].append(pred_arr)
            data["num_gt_dets"] += len(gt_arr)
            data["num_tracker_dets"] += len(pred_arr)
            sim = np.zeros((len(gt_pts), len(pred_pts)))
            if gt_pts and pred_pts:
                masks_by_id = _load_masks_at_frame(gt_masks, fidx, height, width, return_dict=True)
                empty_mask = np.zeros((height, width), dtype=bool)
                gt_oids_ordered = [oid for oid, _ in sorted(gf["points"].items())]
                for i, oid in enumerate(gt_oids_ordered):
                    mask = masks_by_id.get(str(oid), empty_mask)
                    for j, pp in enumerate(pred_pts):
                        if _is_point_in_mask(pp, mask):
                            sim[i, j] = 1.0
            data["similarity_scores"].append(sim)
        return data

    def compute(self, data):
        na = len(self.alpha_thresholds)
        res = {k: np.zeros(na) for k in ["HOTA_TP", "HOTA_FN", "HOTA_FP", "HOTA", "DetA", "AssA", "LocA"]}
        if data["num_tracker_dets"] == 0 and data["num_gt_dets"] == 0:
            return {k: np.ones(na) for k in res}
        if data["num_tracker_dets"] == 0:
            res["HOTA_FN"] = np.full(na, data["num_gt_dets"])
            res["LocA"] = np.ones(na)
            return res
        if data["num_gt_dets"] == 0:
            res["HOTA_FP"] = np.full(na, data["num_tracker_dets"])
            res["LocA"] = np.ones(na)
            return res
        pot_matches = np.zeros((data["num_gt_ids"], data["num_tracker_ids"]))
        gt_count = np.zeros((data["num_gt_ids"], 1))
        pred_count = np.zeros((1, data["num_tracker_ids"]))
        for t, (gids, pids) in enumerate(zip(data["gt_ids"], data["tracker_ids"])):
            sim = data["similarity_scores"][t]
            denom = sim.sum(0)[None, :] + sim.sum(1)[:, None] - sim
            iou = np.zeros_like(sim)
            mask = denom > np.finfo(float).eps
            iou[mask] = sim[mask] / denom[mask]
            pot_matches[gids[:, None], pids[None, :]] += iou
            gt_count[gids] += 1
            pred_count[0, pids] += 1
        global_score = pot_matches / (gt_count + pred_count - pot_matches + np.finfo(float).eps)
        matches_counts = [np.zeros_like(pot_matches) for _ in self.alpha_thresholds]
        for t, (gids, pids) in enumerate(zip(data["gt_ids"], data["tracker_ids"])):
            if len(gids) == 0:
                for a in range(na):
                    res["HOTA_FP"][a] += len(pids)
                continue
            if len(pids) == 0:
                for a in range(na):
                    res["HOTA_FN"][a] += len(gids)
                continue
            sim = data["similarity_scores"][t]
            score_mat = global_score[gids[:, None], pids[None, :]] * sim
            mr, mc = linear_sum_assignment(-score_mat)
            for a, alpha in enumerate(self.alpha_thresholds):
                matched = sim[mr, mc] >= alpha - np.finfo(float).eps
                amr, amc = mr[matched], mc[matched]
                nm = len(amr)
                res["HOTA_TP"][a] += nm
                res["HOTA_FN"][a] += len(gids) - nm
                res["HOTA_FP"][a] += len(pids) - nm
                if nm > 0:
                    res["LocA"][a] += sim[amr, amc].sum()
                    matches_counts[a][gids[amr], pids[amc]] += 1
        for a in range(na):
            mc = matches_counts[a]
            assA = mc / np.maximum(1, gt_count + pred_count - mc)
            res["AssA"][a] = np.sum(mc * assA) / np.maximum(1, res["HOTA_TP"][a])
        res["LocA"] = np.maximum(1e-10, res["LocA"]) / np.maximum(1e-10, res["HOTA_TP"])
        res["DetA"] = res["HOTA_TP"] / np.maximum(1, res["HOTA_TP"] + res["HOTA_FN"] + res["HOTA_FP"])
        res["HOTA"] = np.sqrt(res["DetA"] * res["AssA"])
        return res


def _evaluate_video_object_tracking(pred_tracks, gt_tracks, gt_masks, height, width):
    spatial = _evaluate_video_tracks_with_masks(pred_tracks, gt_tracks, gt_masks, height, width)
    hota = _HOTAMetric()
    prep = hota.prepare_data(pred_tracks, gt_tracks, gt_masks, height, width)
    res = hota.compute(prep)
    alpha_05_idx = 9  # threshold 0.5
    return {
        "precision": spatial["precision"],
        "recall": spatial["recall"],
        "f1": spatial["f1"],
        "HOTA": float(res["HOTA"][alpha_05_idx]),
    }


# ---------------------------------------------------------------------------
# process_results
# ---------------------------------------------------------------------------


def rvos_process_results(doc, results):
    """Parse pred tracks, compute per-query metrics, and store submission record."""
    pred_text = results[0] if results else ""
    width = int(doc["width"])
    height = int(doc["height"])
    video_fps = float(doc["fps"]) if doc["fps"] else 1.0

    pred_tracks = extract_tracks(pred_text, width, height, video_fps)
    try:
        gt_tracks = json.loads(doc["frame_trajectories"])
        # Normalize GT points dict-of-obj shape
        for entry in gt_tracks:
            if isinstance(entry.get("points"), list):
                entry["points"] = {
                    str(p["id"]): {"point": p["point"], "occluded": p.get("occluded", False)}
                    for p in entry["points"]
                }
    except Exception:
        gt_tracks = []
    try:
        gt_masks = _load_gt_masks(doc)
    except Exception as e:
        eval_logger.warning(f"[rvos] Failed to load GT masks for {doc.get('id')}: {e}")
        gt_masks = {}

    try:
        metrics = _evaluate_video_object_tracking(pred_tracks, gt_tracks, gt_masks, height, width)
    except Exception as e:
        eval_logger.warning(f"[rvos] Metric computation failed for {doc.get('id')}: {e}")
        metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "HOTA": 0.0}

    record = {
        "id": doc["id"],
        "qid": doc.get("qid"),
        "video": doc["video"],
        "expression": doc["expression"],
        "prediction": pred_text,
        "tracks": pred_tracks,
        "height": height,
        "width": width,
        "fps": int(doc["fps"]),
        "sampling_fps": int(doc.get("sampling_fps", 1)),
        "n_frames": int(doc["n_frames"]),
        "mask_ids": list(doc["mask_ids"]),
        **metrics,
    }
    return {
        "rvos_f1": record,
        "rvos_precision": record,
        "rvos_recall": record,
        "rvos_hota": record,
    }


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------


def _dump_submission(records: List[Dict[str, Any]]) -> str:
    task = records[0]["id"].split("_track_")[0] if records else "rvos"
    output_dir = os.environ.get("RVOS_OUTPUT_DIR", "rvos_output")
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{task}_submission.json")
    with open(path, "w") as f:
        json.dump(records, f)
    return path


_MEAN = {"f1": None, "precision": None, "recall": None, "HOTA": None}


def _mean_metric(records: List[Dict[str, Any]], key: str) -> float:
    if not records:
        return 0.0
    return float(np.mean([r.get(key, 0.0) for r in records]))


def rvos_aggregate_f1(results):
    if results:
        path = _dump_submission(results)
        eval_logger.info(f"[rvos] Submission saved to {path} ({len(results)} predictions).")
    return _mean_metric(results, "f1")


def rvos_aggregate_precision(results):
    return _mean_metric(results, "precision")


def rvos_aggregate_recall(results):
    return _mean_metric(results, "recall")


def rvos_aggregate_hota(results):
    return _mean_metric(results, "HOTA")
