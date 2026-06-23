# ExtremeWhenBench
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""ExtremeWhenBench: hour-scale natural-language temporal grounding.

Videos are sourced from LVBench / MLVU / VideoMME and are NOT redistributed
with this task. ``ewb_doc_to_visual`` locates each video in the lmms-eval
cache for its source corpus, with optional env-var overrides.
"""

import os
import re
import sys
from typing import Any, Optional

from loguru import logger as eval_logger

HF_HOME = os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface"))

CORPUS_CACHE = {
    "LVBench": os.path.join(HF_HOME, "lvbench"),
    "MLVU": os.path.join(HF_HOME, "mlvu"),
    "VideoMME": os.path.join(HF_HOME, "videomme", "data"),
}

EXTS = (".mp4", ".MP4", ".mkv", ".webm", ".avi")

_TIME_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:-|to|~|–|—)\s*(\d+(?:\.\d+)?)")


def _candidate_paths(corpus: str, vid: str) -> list[str]:
    """Build an ordered list of plausible video paths for a corpus/video-id.

    EWB_PREEXTRACTED_DIR is checked first: if set, point it at a directory of
    pre-extracted lightweight videos (e.g., 1024-frame, 384x re-encodes named
    ``{video_id}.mp4``). This skips the full hour-long source decode and
    typically gives a ~10x speedup with no metric change at N<=1024.
    """
    bases: list[str] = []
    preextracted = os.getenv("EWB_PREEXTRACTED_DIR")
    if preextracted:
        bases.append(preextracted)
    override = os.getenv(f"EWB_{corpus.upper()}_PATH")
    if override:
        bases.append(override)
    bases.append(CORPUS_CACHE[corpus])

    paths: list[str] = []
    for base in bases:
        for ext in EXTS:
            paths.append(os.path.join(base, vid + ext))
        if os.path.isdir(base):
            for sub in os.listdir(base):
                sub_path = os.path.join(base, sub)
                if not os.path.isdir(sub_path):
                    continue
                for ext in EXTS:
                    paths.append(os.path.join(sub_path, vid + ext))
    return paths


_DOWNLOAD_HINT = {
    "LVBench": "python -c \"from datasets import load_dataset; load_dataset('lmms-lab/LVBench')\"",
    "MLVU": "python -c \"from datasets import load_dataset; load_dataset('sy1998/MLVU_dev')\"",
    "VideoMME": "python -c \"from datasets import load_dataset; load_dataset('lmms-lab/Video-MME')\"",
}


def ewb_process_docs(dataset):
    """Preflight: verify every source-corpus video is resolvable before any inference.

    Aggregates missing items across all three corpora and raises a single
    actionable error, so the user does not hit a FileNotFoundError several
    hours into a full run after only one corpus was downloaded.
    """
    missing: dict[str, list[str]] = {}
    seen: set[tuple[str, str]] = set()
    for doc in dataset:
        key = (doc["source_corpus"], doc["video_id"])
        if key in seen:
            continue
        seen.add(key)
        corpus, vid = key
        if not any(os.path.exists(p) for p in _candidate_paths(corpus, vid)):
            missing.setdefault(corpus, []).append(vid)
    if missing:
        lines = [
            "",
            "=" * 70,
            "ExtremeWhenBench preflight: video files missing for the corpora below.",
            "=" * 70,
        ]
        for corpus in sorted(missing):
            vids = missing[corpus]
            lines.append(f"")
            lines.append(f"  [{corpus}] {len(vids)} video(s) missing under {CORPUS_CACHE[corpus]}")
            lines.append(f"    Download:  {_DOWNLOAD_HINT[corpus]}")
            lines.append(f"    Or set:    export EWB_{corpus.upper()}_PATH=/abs/path/to/{corpus.lower()}/videos")
        lines.append("")
        lines.append("See lmms_eval/tasks/extremewhenbench/README.md for the full setup guide.")
        lines.append("=" * 70)
        # Print + SystemExit so the message is visible — tenacity's retry on
        # task.download() catches normal exceptions and hides the body.
        print("\n".join(lines), file=sys.stderr, flush=True)
        sys.exit(1)
    return dataset


def ewb_doc_to_visual(doc: dict[str, Any]) -> list[str]:
    corpus = doc["source_corpus"]
    vid = doc["video_id"]
    for p in _candidate_paths(corpus, vid):
        if os.path.exists(p):
            return [p]
    # process_docs preflight should have caught this already; fall back to a
    # per-doc error if the task is invoked without the preflight (e.g., when
    # users disable process_docs).
    raise FileNotFoundError(
        f"Video for qid={doc['qid']} ({corpus}/{vid}) not found. "
        f"Download via:\n  {_DOWNLOAD_HINT[corpus]}\n"
        f"or set EWB_{corpus.upper()}_PATH to a directory containing {vid}.mp4."
    )


DEFAULT_PROMPT = (
    "[Video Temporal Grounding]\n"
    "You are watching a {dur:.0f}-second long video.\n"
    "Please find the moment described by the following question, "
    "determining its starting and ending times. The format should be: "
    "'The event happens in the start time - end time'. "
    "For example, The event 'person turn a light on' happens in the "
    "24.3 - 30.4 seconds. "
    'Now I will give you the question: "{question}"\n'
    "Please return its start time and end time."
)


def ewb_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs: Optional[dict] = None) -> str:
    lmms_eval_specific_kwargs = lmms_eval_specific_kwargs or {}
    template = lmms_eval_specific_kwargs.get("prompt_template", DEFAULT_PROMPT)
    dur = float(doc.get("video_duration_s") or 0)
    return template.format(dur=dur, question=doc["question"])


def ewb_doc_to_target(doc: dict[str, Any]) -> list[float]:
    return [float(x) for x in doc["correct_interval"]]


def _parse_interval(text: str) -> Optional[tuple[float, float]]:
    matches = _TIME_RE.findall(text)
    if not matches:
        return None
    s, e = matches[-1]
    s_f, e_f = float(s), float(e)
    if s_f > e_f:
        s_f, e_f = e_f, s_f
    return s_f, e_f


def _iou(pred: tuple[float, float], gt: tuple[float, float]) -> float:
    inter = max(0.0, min(pred[1], gt[1]) - max(pred[0], gt[0]))
    union = max(pred[1], gt[1]) - min(pred[0], gt[0])
    return inter / union if union > 0 else 0.0


def ewb_process_results(doc: dict[str, Any], results: list[str]) -> dict[str, dict[str, Any]]:
    pred_text = results[0] if results else ""
    interval = _parse_interval(pred_text)
    gt = (float(doc["correct_interval"][0]), float(doc["correct_interval"][1]))
    if interval is None:
        iou_val = 0.0
        parsed = False
    else:
        iou_val = _iou(interval, gt)
        parsed = True

    rec = {
        "qid": doc["qid"],
        "video_id": doc["video_id"],
        "source_corpus": doc["source_corpus"],
        "pred_raw": pred_text,
        "pred_interval": list(interval) if interval else None,
        "gt": list(gt),
        "iou": iou_val,
        "parsed": parsed,
    }
    return {
        "ewb_mIoU": rec,
        "ewb_R@0.3": rec,
        "ewb_R@0.5": rec,
        "ewb_R@0.7": rec,
        "ewb_parse_fail": rec,
    }


def ewb_aggregate_miou(records: list[dict[str, Any]]) -> float:
    if not records:
        return 0.0
    return sum(r["iou"] for r in records) / len(records)


def _recall_at(records: list[dict[str, Any]], tau: float) -> float:
    if not records:
        return 0.0
    return sum(1 for r in records if r["iou"] >= tau) / len(records)


def ewb_aggregate_r03(records: list[dict[str, Any]]) -> float:
    return _recall_at(records, 0.3)


def ewb_aggregate_r05(records: list[dict[str, Any]]) -> float:
    return _recall_at(records, 0.5)


def ewb_aggregate_r07(records: list[dict[str, Any]]) -> float:
    return _recall_at(records, 0.7)


def ewb_aggregate_parse_fail(records: list[dict[str, Any]]) -> float:
    if not records:
        return 0.0
    return sum(1 for r in records if not r["parsed"]) / len(records)
