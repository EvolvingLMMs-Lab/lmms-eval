"""VBVR-Bench (Image-to-Video reasoning) utilities for lmms-eval.

Evaluation flow
---------------
1. Each sample carries a first-frame image (base64 PNG) + a text prompt.
2. The model generates an MP4 and returns JSON: ``{"text": "", "videos": [path]}``.
3. ``vbvr_process_results`` parses the JSON, resolves the matching ground-truth
   folder under ``$VBVR_GT_PATH``, dispatches to the per-task rule-based
   evaluator from the vendored ``vbvr_bench`` package, and records a per-sample
   score + dimension breakdown.
4. Aggregation functions compute In-Domain / Out-of-Domain / per-category means
   and an overall mean matching the upstream VBVRBench output.

Environment variables
---------------------
- ``VBVR_GT_PATH``: local root of the downloaded Video-Reason/VBVR-Bench-Data
  dataset. Must contain ``In-Domain_50/`` and ``Out-of-Domain_50/`` folders with
  ``{task_name}/{video_idx}/{first_frame.png,final_frame.png,ground_truth.mp4,prompt.txt}``.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from loguru import logger as eval_logger
from PIL import Image

from lmms_eval.tasks.vbvr.vbvr_bench.evaluators import (
    get_evaluator,
    get_split,
    get_task_category,
)

CATEGORIES = ("Abstraction", "Knowledge", "Perception", "Spatiality", "Transformation")
SPLITS = ("In_Domain", "Out_of_Domain")

_BASE64_PREFIX = re.compile(r"^data:image/[^;]+;base64,", re.IGNORECASE)


def _gt_root() -> str:
    root = os.getenv("VBVR_GT_PATH")
    if not root:
        raise RuntimeError(
            "VBVR_GT_PATH is not set. Download the GT with:\n"
            "    hf download Video-Reason/VBVR-Bench-Data --repo-type dataset --local-dir <path>\n"
            "then `export VBVR_GT_PATH=<path>`."
        )
    return os.path.expanduser(os.path.expandvars(root))


def _decode_base64_image(data: str) -> Image.Image:
    payload = _BASE64_PREFIX.sub("", data)
    return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")


def _parse_task_meta(doc: Dict[str, Any]) -> Tuple[str, str, str]:
    """Return (file_split, task_name, video_idx) parsed from first_frame_path.

    Example path:
        In-Domain_50/G-131_select_next_figure_..._data-generator/00000/first_frame.png
    """
    raw = doc.get("first_frame_path") or doc.get("final_frame_path") or doc.get("prompt_path") or doc.get("ground_truth_video_path") or ""
    parts = [p for p in str(raw).split("/") if p]
    if len(parts) < 3:
        raise ValueError(f"Cannot parse VBVR task meta from path: {raw!r}")
    file_split, task_name, video_idx = parts[0], parts[1], parts[2]
    return file_split, task_name, video_idx


def vbvr_doc_to_visual(doc: Dict[str, Any]) -> List[Image.Image]:
    first_image = doc.get("first_image")
    if not first_image:
        return []
    return [_decode_base64_image(first_image)]


def vbvr_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> str:
    prompt = str(doc.get("prompt") or "").strip()
    pre_prompt = ""
    post_prompt = ""
    if lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    return f"{pre_prompt}{prompt}{post_prompt}"


def vbvr_doc_to_target(doc: Dict[str, Any]) -> str:
    return str(doc.get("prompt") or "")


def vbvr_doc_to_messages(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    visuals = vbvr_doc_to_visual(doc)
    text = vbvr_doc_to_text(doc, lmms_eval_specific_kwargs)
    content: List[Dict[str, Any]] = [{"type": "image", "url": img} for img in visuals]
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


def _load_prediction(results: Sequence[Any]) -> Dict[str, Any]:
    pred = results[0] if results else "{}"
    if isinstance(pred, dict):
        return pred
    try:
        return json.loads(pred)
    except (json.JSONDecodeError, TypeError):
        eval_logger.warning(f"Failed to parse VBVR prediction JSON: {str(pred)[:200]}")
        return {}


def _video_from_prediction(pred: Dict[str, Any]) -> Optional[str]:
    videos = pred.get("videos", [])
    if not isinstance(videos, (list, tuple)) or not videos:
        return None
    path = os.path.expanduser(os.path.expandvars(str(videos[0]).strip()))
    if not path or not os.path.isfile(path):
        eval_logger.warning(f"VBVR generated video path missing: {path}")
        return None
    return os.path.abspath(path)


def _build_eval_info(doc: Dict[str, Any], video_path: str, file_split: str, task_name: str, video_idx: str) -> Dict[str, Any]:
    gt_root = _gt_root()
    gt_task_dir = os.path.join(gt_root, file_split, task_name, video_idx)
    return {
        "video_path": video_path,
        "gt_path": gt_task_dir,
        "gt_video_path": os.path.join(gt_task_dir, "ground_truth.mp4"),
        "gt_first_frame": os.path.join(gt_task_dir, "first_frame.png"),
        "gt_final_frame": os.path.join(gt_task_dir, "final_frame.png"),
        "prompt_path": os.path.join(gt_task_dir, "prompt.txt"),
        "prompt": str(doc.get("prompt") or ""),
        "task_name": task_name,
        "split": get_split(task_name),
        "file_split": file_split,
        "video_idx": video_idx,
    }


def _empty_entry(doc: Dict[str, Any], status: str, video_path: Optional[str] = None) -> Dict[str, Any]:
    try:
        file_split, task_name, video_idx = _parse_task_meta(doc)
    except Exception:
        file_split, task_name, video_idx = "", "", ""
    return {
        "task_name": task_name,
        "file_split": file_split,
        "video_idx": video_idx,
        "split": get_split(task_name) if task_name else "Unknown",
        "category": get_task_category(task_name) if task_name else "Unknown",
        "score": 0.0,
        "dimensions": {},
        "video_path": video_path,
        "status": status,
        "valid": False,
    }


def _fanout_metrics(entry: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Emit the same entry to every metric so each aggregator can filter."""
    return {
        "vbvr_overall": entry,
        "vbvr_in_domain": entry,
        "vbvr_out_of_domain": entry,
        "vbvr_abstraction": entry,
        "vbvr_knowledge": entry,
        "vbvr_perception": entry,
        "vbvr_spatiality": entry,
        "vbvr_transformation": entry,
    }


def vbvr_process_results(doc: Dict[str, Any], results: Sequence[Any], **kwargs) -> Dict[str, Any]:
    pred = _load_prediction(results)
    video_path = _video_from_prediction(pred)

    if not video_path:
        return _fanout_metrics(_empty_entry(doc, status="missing_video"))

    try:
        file_split, task_name, video_idx = _parse_task_meta(doc)
    except Exception as e:
        eval_logger.error(f"VBVR: failed to parse task meta: {e}")
        return _fanout_metrics(_empty_entry(doc, status="bad_meta", video_path=video_path))

    eval_info = _build_eval_info(doc, video_path, file_split, task_name, video_idx)

    try:
        evaluator = get_evaluator(task_name)
        vr = evaluator.evaluate(eval_info)
        score = float(vr.get("score", 0.0))
        dimensions = {k: float(v) for k, v in vr.get("dimensions", {}).items() if isinstance(v, (int, float))}
        status = "error" if "error" in vr else "ok"
    except Exception as e:
        eval_logger.error(f"VBVR evaluator failed for {task_name}/{video_idx}: {str(e)[:300]}")
        score = 0.0
        dimensions = {}
        status = f"evaluator_error"

    entry = {
        "task_name": task_name,
        "file_split": file_split,
        "video_idx": video_idx,
        "split": get_split(task_name),
        "category": get_task_category(task_name),
        "score": score,
        "dimensions": dimensions,
        "video_path": video_path,
        "status": status,
        "valid": status == "ok",
    }
    return _fanout_metrics(entry)


def _entries(results: Iterable[Any]) -> List[Dict[str, Any]]:
    out = []
    for r in results:
        if isinstance(r, dict) and isinstance(r.get("score"), (int, float)):
            out.append(r)
    return out


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _agg_by(results, key: str, value: str, label: str) -> float:
    entries = _entries(results)
    selected = [e for e in entries if e.get(key) == value]
    if not selected:
        eval_logger.warning(f"[VBVR] No samples for {label}.")
        return 0.0
    mean = _mean([e["score"] for e in selected])
    eval_logger.info(f"[VBVR] {label}: {mean:.4f} (n={len(selected)})")
    return mean


def vbvr_aggregate_overall(results) -> float:
    entries = _entries(results)
    if not entries:
        return 0.0
    in_dom = [e["score"] for e in entries if e.get("split") == "In_Domain"]
    ood = [e["score"] for e in entries if e.get("split") == "Out_of_Domain"]
    # Weighted by sample count — matches upstream VBVRBench.
    total = len(in_dom) + len(ood)
    if total == 0:
        return 0.0
    mean = (_mean(in_dom) * len(in_dom) + _mean(ood) * len(ood)) / total
    eval_logger.info(
        f"[VBVR] Overall: {mean:.4f} "
        f"(In-Domain {_mean(in_dom):.4f} n={len(in_dom)}, "
        f"Out-of-Domain {_mean(ood):.4f} n={len(ood)})"
    )
    # Per-category breakdown for the logs
    cat_scores: Dict[str, List[float]] = defaultdict(list)
    for e in entries:
        cat_scores[e.get("category", "Unknown")].append(e["score"])
    for cat in CATEGORIES:
        if cat_scores.get(cat):
            eval_logger.info(f"[VBVR] {cat}: {_mean(cat_scores[cat]):.4f} (n={len(cat_scores[cat])})")
    return mean


def vbvr_aggregate_in_domain(results) -> float:
    return _agg_by(results, "split", "In_Domain", "In-Domain")


def vbvr_aggregate_out_of_domain(results) -> float:
    return _agg_by(results, "split", "Out_of_Domain", "Out-of-Domain")


def vbvr_aggregate_abstraction(results) -> float:
    return _agg_by(results, "category", "Abstraction", "Abstraction")


def vbvr_aggregate_knowledge(results) -> float:
    return _agg_by(results, "category", "Knowledge", "Knowledge")


def vbvr_aggregate_perception(results) -> float:
    return _agg_by(results, "category", "Perception", "Perception")


def vbvr_aggregate_spatiality(results) -> float:
    return _agg_by(results, "category", "Spatiality", "Spatiality")


def vbvr_aggregate_transformation(results) -> float:
    return _agg_by(results, "category", "Transformation", "Transformation")


def _is_in_domain_row(row: Dict[str, Any]) -> bool:
    try:
        _, task_name, _ = _parse_task_meta(row)
    except Exception:
        return False
    return get_split(task_name) == "In_Domain"


def _is_out_of_domain_row(row: Dict[str, Any]) -> bool:
    try:
        _, task_name, _ = _parse_task_meta(row)
    except Exception:
        return False
    return get_split(task_name) == "Out_of_Domain"


def filter_in_domain(dataset):
    return dataset.filter(_is_in_domain_row)


def filter_out_of_domain(dataset):
    return dataset.filter(_is_out_of_domain_row)
