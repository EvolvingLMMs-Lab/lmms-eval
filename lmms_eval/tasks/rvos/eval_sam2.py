"""Stage-2 evaluation for RVOS tasks: SAM2 mask propagation + J&F metrics.

Consumes the ``<task>_submission.json`` file produced by stage-1 (the
lmms-eval ``rvos_*`` tasks) and computes mask-level J / F / J&F / HOTA by
propagating each query's predicted point tracks through a video with a
SAM2 video predictor from ``facebook/sam2.1-hiera-large``.

Requires:  ``pip install sam2 pycocotools scikit-image opencv-python``

Usage (single GPU)::

    python -m lmms_eval.tasks.rvos.eval_sam2 \\
        --task rvos_ref_davis17 \\
        --predictions rvos_output/ref-davis17_submission.json \\
        --output rvos_sam2_output

Multi-GPU (torchrun)::

    torchrun --nproc-per-node 8 -m lmms_eval.tasks.rvos.eval_sam2 \\
        --task rvos_ref_davis17 \\
        --predictions rvos_output/ref-davis17_submission.json

Each rank processes a disjoint slice of the submission; per-query results are
merged on rank 0 and written to ``<output>/<task>_sam2_metrics.json`` and
``<output>/<task>_sam2_predictions.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from collections import defaultdict
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from loguru import logger as eval_logger

TASK_TO_CACHE = {
    "rvos_ref_davis17": "rvos_ref_davis17",
    "rvos_mevis_valid_u": "rvos_mevis_valid_u",
    "rvos_ref_yt_vos": "rvos_ref_yt_vos",
    "rvos_reasonvos": "rvos_reasonvos",
}

DEFAULT_SAM2_MODEL = "facebook/sam2.1-hiera-large"
ALPHA_THRESHOLDS = np.arange(0.05, 1.0, 0.05)


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------


def setup_distributed() -> tuple[int, int, torch.device]:
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=120))
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        return rank, world, torch.device(f"cuda:{local_rank}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return 0, 1, device


def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


# ---------------------------------------------------------------------------
# SAM2 loading via official API
# ---------------------------------------------------------------------------


def build_sam2_predictor(model_id: str, device: torch.device):
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    predictor = SAM2VideoPredictor.from_pretrained(model_id)
    predictor.to(device)
    predictor.eval()
    return predictor


def _mask_logits_to_uint8(mask_logits: torch.Tensor) -> torch.Tensor:
    """Reduce per-object mask logits to a single binary uint8 mask (union)."""
    m = mask_logits.detach()
    if m.ndim == 4:
        m = m.squeeze(1)  # (n_obj, 1, H, W) -> (n_obj, H, W)
    if m.ndim == 3 and m.shape[0] > 0:
        m, _ = m.max(dim=0)
    return (m > 0).to(torch.uint8).cpu()


def run_sam2(predictor, prompt_map: Dict[tuple, List[tuple]], inference_state) -> np.ndarray:
    """Propagate points through the video. Returns pred masks (T, H, W) uint8."""
    num_frames = int(inference_state["num_frames"])
    video_h = int(inference_state["video_height"])
    video_w = int(inference_state["video_width"])

    frame_to_mask: Dict[int, torch.Tensor] = {}
    prompts_by_frame = defaultdict(list)
    for (frame_idx, obj_id), pts in prompt_map.items():
        prompts_by_frame[frame_idx].append((obj_id, pts))
    prompt_frames = sorted(prompts_by_frame)

    if prompt_frames:
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if torch.cuda.is_available() and "cuda" in str(inference_state["device"]) else nullcontext()
        with torch.inference_mode(), autocast_ctx:
            for i, fidx in enumerate(prompt_frames):
                for obj_id, points in prompts_by_frame[fidx]:
                    pts = torch.tensor(points, dtype=torch.float32)
                    labels = torch.ones((len(points),), dtype=torch.int32)
                    predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=fidx,
                        obj_id=obj_id,
                        points=pts,
                        labels=labels,
                        clear_old_points=True,
                        normalize_coords=True,
                    )
                next_pf = prompt_frames[i + 1] if i + 1 < len(prompt_frames) else num_frames
                max_track = max(next_pf - fidx - 1, 0)
                for out_idx, _, out_logits in predictor.propagate_in_video(
                    inference_state=inference_state,
                    start_frame_idx=fidx,
                    max_frame_num_to_track=max_track,
                    reverse=False,
                ):
                    frame_to_mask[int(out_idx)] = _mask_logits_to_uint8(out_logits)
            first = prompt_frames[0]
            if first > 0:
                for out_idx, _, out_logits in predictor.propagate_in_video(
                    inference_state=inference_state,
                    start_frame_idx=first,
                    max_frame_num_to_track=first,
                    reverse=True,
                ):
                    frame_to_mask[int(out_idx)] = _mask_logits_to_uint8(out_logits)

    empty = torch.zeros((video_h, video_w), dtype=torch.uint8)
    return np.stack([frame_to_mask.get(i, empty).numpy() for i in range(num_frames)])


# ---------------------------------------------------------------------------
# Prompt map + frame extraction
# ---------------------------------------------------------------------------


def build_prompt_map(tracks: List[Dict[str, Any]], W: int, H: int, video_fps: float, num_frames: int) -> Dict[tuple, List[tuple]]:
    prompt_map: Dict[tuple, List[tuple]] = defaultdict(list)
    for entry in tracks:
        ts = float(entry.get("time", entry.get("frame", 0) / max(video_fps, 1e-6)))
        frame_idx = int(round(ts * video_fps))
        frame_idx = max(0, min(frame_idx, num_frames - 1))
        for oid, p in entry.get("points", {}).items():
            xy = p["point"] if isinstance(p, dict) else p
            x = max(0.0, min(float(xy[0]), float(W - 1)))
            y = max(0.0, min(float(xy[1]), float(H - 1)))
            prompt_map[(frame_idx, str(oid))].append((x, y))
    return dict(prompt_map)


def extract_frames_ffmpeg(video_path: str, out_dir: str, fps: float) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", video_path, "-vf", f"fps={fps}", os.path.join(out_dir, "%05d.jpg")],
        check=True,
    )


# ---------------------------------------------------------------------------
# GT mask loading + metrics (J / F / J&F / HOTA)
# ---------------------------------------------------------------------------


def load_gt_masks(cache_root: str, video: str, mask_ids: List[str], num_frames: int, H: int, W: int) -> np.ndarray:
    import cv2
    from pycocotools import mask as mask_utils

    gt = np.zeros((num_frames, H, W), dtype=np.uint8)
    for mid in mask_ids:
        path = os.path.join(cache_root, "masks", video, f"{mid}.json")
        if not os.path.exists(path):
            eval_logger.warning(f"GT mask file missing: {path}")
            continue
        with open(path, "r") as f:
            data = json.load(f)
        for _, rle_list in data.items():
            for fi, rle in enumerate(rle_list):
                if fi >= num_frames or rle is None:
                    continue
                decoded = mask_utils.decode(rle)
                if decoded.shape != (H, W):
                    decoded = cv2.resize(decoded, (W, H), interpolation=cv2.INTER_NEAREST)
                gt[fi] = np.maximum(gt[fi], decoded)
    return gt


def _seg2bmap(seg: np.ndarray) -> np.ndarray:
    seg = seg.astype(bool)
    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)
    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]
    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0
    return b


def f_measure(fg: np.ndarray, gt: np.ndarray, bound_th: float = 0.008) -> float:
    import cv2
    from skimage.morphology import disk

    bound_pix = bound_th if bound_th >= 1 else np.ceil(bound_th * np.linalg.norm(fg.shape))
    fg_b = _seg2bmap(fg)
    gt_b = _seg2bmap(gt)
    disk_k = disk(bound_pix).astype(np.uint8)
    fg_dil = cv2.dilate(fg_b.astype(np.uint8), disk_k)
    gt_dil = cv2.dilate(gt_b.astype(np.uint8), disk_k)
    gt_match = gt_b * fg_dil
    fg_match = fg_b * gt_dil
    n_fg = np.sum(fg_b)
    n_gt = np.sum(gt_b)
    if n_fg == 0 and n_gt > 0:
        p, r = 1.0, 0.0
    elif n_fg > 0 and n_gt == 0:
        p, r = 0.0, 1.0
    elif n_fg == 0 and n_gt == 0:
        p, r = 1.0, 1.0
    else:
        p = float(fg_match.sum()) / float(n_fg)
        r = float(gt_match.sum()) / float(n_gt)
    return 0.0 if (p + r) == 0 else float(2 * p * r / (p + r))


def db_eval_iou(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    inter = np.sum(gt & pred, axis=(-2, -1))
    union = np.sum(gt | pred, axis=(-2, -1))
    j = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
    return j.astype(float)


def db_eval_boundary(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    n = gt.shape[0]
    out = np.zeros(n)
    for i in range(n):
        out[i] = f_measure(pred[i], gt[i])
    return out


def compute_hota_single(gt: np.ndarray, pred: np.ndarray) -> float:
    n = gt.shape[0]
    ious = np.zeros(n)
    for t in range(n):
        inter = np.logical_and(gt[t] > 0, pred[t] > 0).sum()
        union = np.logical_or(gt[t] > 0, pred[t] > 0).sum()
        ious[t] = inter / union if union > 0 else 1.0
    gt_ex = np.array([gt[t].sum() > 0 for t in range(n)])
    pd_ex = np.array([pred[t].sum() > 0 for t in range(n)])
    hs = np.zeros(len(ALPHA_THRESHOLDS))
    for a, alpha in enumerate(ALPHA_THRESHOLDS):
        tp = fn = fp = 0
        for t in range(n):
            gt_here, pd_here = gt_ex[t], pd_ex[t]
            if not gt_here and not pd_here:
                continue
            if gt_here and pd_here and ious[t] >= alpha:
                tp += 1
            else:
                if gt_here:
                    fn += 1
                if pd_here:
                    fp += 1
        det_a = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 1.0
        hs[a] = np.sqrt(det_a)
    return float(np.mean(hs))


def evaluate_masks(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    j_arr = db_eval_iou(gt, pred)
    f_arr = db_eval_boundary(gt, pred)
    j = float(j_arr.mean()) if j_arr.size else 0.0
    f = float(f_arr.mean()) if f_arr.size else 0.0
    return {"J": j, "F": f, "J&F": (j + f) / 2.0, "HOTA": compute_hota_single(gt, pred)}


# ---------------------------------------------------------------------------
# Per-query processing
# ---------------------------------------------------------------------------


def process_query(predictor, record: Dict[str, Any], cache_root: str, tmp_root: Optional[str], video_fps: float, device: torch.device) -> Optional[Dict[str, Any]]:
    video = record["video"]
    video_path = os.path.join(cache_root, "videos", f"{video}.mp4")
    if not os.path.exists(video_path):
        eval_logger.warning(f"[sam2] video not found: {video_path}")
        return None

    H, W = int(record["height"]), int(record["width"])
    tracks = record.get("tracks") or []

    with tempfile.TemporaryDirectory(prefix=f"rvos_{video}_", dir=tmp_root) as td:
        extract_frames_ffmpeg(video_path, td, video_fps)
        state = predictor.init_state(video_path=td, offload_video_to_cpu=True)
        state["device"] = device

        prompt_map = build_prompt_map(tracks, W, H, video_fps, state["num_frames"])
        if not prompt_map:
            pred_masks = np.zeros((state["num_frames"], state["video_height"], state["video_width"]), dtype=np.uint8)
        else:
            pred_masks = run_sam2(predictor, prompt_map, state)
        gt_masks = load_gt_masks(cache_root, video, list(record.get("mask_ids") or []), pred_masks.shape[0], pred_masks.shape[1], pred_masks.shape[2])

    metrics = evaluate_masks(gt_masks, pred_masks)
    return {
        "id": record.get("id"),
        "qid": record.get("qid"),
        "video": video,
        "expression": record.get("expression"),
        "num_prompt_frames": len({fi for fi, _ in prompt_map}),
        "num_prompt_points": sum(len(v) for v in prompt_map.values()),
        **metrics,
    }


def gather_records(records: List[Dict[str, Any]], world: int) -> List[Dict[str, Any]]:
    if world == 1:
        return records
    obj_list: List[List[Dict[str, Any]]] = [[] for _ in range(world)]
    dist.all_gather_object(obj_list, records)
    merged: List[Dict[str, Any]] = []
    for lst in obj_list:
        merged.extend(lst)
    return merged


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASK_TO_CACHE.keys()))
    ap.add_argument("--predictions", required=True, help="Path to stage-1 <task>_submission.json")
    ap.add_argument("--sam2-model", default=DEFAULT_SAM2_MODEL, help="HF model id (default: facebook/sam2.1-hiera-large)")
    ap.add_argument("--output", default="rvos_sam2_output")
    ap.add_argument("--cache-dir", default=None, help="Override <hf_home>/<task cache dir>")
    ap.add_argument("--video-fps", type=float, default=6.0)
    ap.add_argument("--tmp-root", default=None)
    ap.add_argument("--max-items", type=int, default=-1)
    args = ap.parse_args()

    rank, world, device = setup_distributed()

    if args.cache_dir is None:
        hf_home = os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface/"))
        cache_root = os.path.join(hf_home, TASK_TO_CACHE[args.task])
    else:
        cache_root = args.cache_dir
    if not os.path.isdir(cache_root):
        raise FileNotFoundError(f"Cache dir missing (run stage 1 first?): {cache_root}")

    with open(args.predictions, "r") as f:
        records = json.load(f)
    if args.max_items > 0:
        records = records[: args.max_items]

    if is_main():
        eval_logger.info(f"[sam2] task={args.task} world={world} total={len(records)} cache={cache_root} model={args.sam2_model}")
    barrier()

    predictor = build_sam2_predictor(args.sam2_model, device)

    local_records = records[rank::world]
    local_results: List[Dict[str, Any]] = []
    progress_every = max(1, len(local_records) // 20)
    for i, rec in enumerate(local_records):
        try:
            r = process_query(predictor, rec, cache_root, args.tmp_root, args.video_fps, device)
        except Exception as e:
            eval_logger.warning(f"[sam2] rank {rank} record {rec.get('id')} failed: {e}")
            r = None
        if r is not None:
            local_results.append(r)
        if is_main() and (i + 1) % progress_every == 0:
            eval_logger.info(f"[sam2] rank 0 progress: {i + 1}/{len(local_records)}")

    all_results = gather_records(local_results, world)
    if not is_main():
        return

    os.makedirs(args.output, exist_ok=True)
    per_query_path = os.path.join(args.output, f"{args.task}_sam2_predictions.json")
    with open(per_query_path, "w") as f:
        json.dump(all_results, f)

    if all_results:
        j = float(np.mean([r["J"] for r in all_results]))
        f_ = float(np.mean([r["F"] for r in all_results]))
        jf = float(np.mean([r["J&F"] for r in all_results]))
        hota = float(np.mean([r["HOTA"] for r in all_results]))
    else:
        j = f_ = jf = hota = 0.0
    summary = {"task": args.task, "n": len(all_results), "J": j, "F": f_, "J&F": jf, "HOTA": hota}
    with open(os.path.join(args.output, f"{args.task}_sam2_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    eval_logger.info(f"[sam2] === {args.task}  N={summary['n']} ===")
    eval_logger.info(f"[sam2]   J   = {j:.4f}")
    eval_logger.info(f"[sam2]   F   = {f_:.4f}")
    eval_logger.info(f"[sam2]   J&F = {jf:.4f}")
    eval_logger.info(f"[sam2]   HOTA= {hota:.4f}")


if __name__ == "__main__":
    main()
