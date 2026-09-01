"""Minimal Video Pairs (MVP) task for lmms-eval.

A shortcut-aware binary VideoQA benchmark with minimally-different
video pairs covering four sub-domains:

  - human_object_interactions
  - robot_object_interactions
  - intuitive_physics
  - temporal_reasoning

Each sub-domain has two splits: ``full`` (the real eval set) and
``mini`` (a smaller balanced subset for faster iteration). The task
``mvp`` groups the four ``full`` sub-tasks; ``mvp_mini`` groups the
four ``mini`` sub-tasks.

Dataset: facebook/minimal_video_pairs (parquet QA only — videos are
NOT redistributed for licensing reasons and must be set up locally
from the 9 source datasets per
https://github.com/facebookresearch/minimal_video_pairs).

The local video root is configured via the ``MVP_VIDEO_DIR`` env var,
which should point at a directory containing one subdirectory per
source (pt, ssv2, language_table, intphys, inflevel, grasp, clevrer,
star, vinoground).

Metrics:
  - ``single_accuracy``: per-clip accuracy
  - ``pair_accuracy``: pair-level accuracy (both clips in a minimally
     different pair must be correct)
"""

from __future__ import annotations

import ast
import os
import re
from typing import Any, Dict, List, Union

from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.default_template_yaml import load_default_template_yaml

IDX_MAP = ["a", "b"]

config = load_default_template_yaml(__file__)

_SETUP_HINT = (
    "MVP video files are not redistributed by facebook/minimal_video_pairs for "
    "licensing reasons. Follow the setup at "
    "https://github.com/facebookresearch/minimal_video_pairs to download the 9 "
    "source datasets, then place them under <HF_HOME>/<metadata.video_cache_dir> "
    "(or point the MVP_VIDEO_DIR env var at their directory) — one sub-directory "
    "per source: pt, ssv2, language_table, intphys, inflevel, grasp, clevrer, star, "
    "vinoground."
)


def _video_root() -> str:
    # The MVP_VIDEO_DIR env var, if set, is an absolute override. Otherwise the
    # video root is metadata.video_cache_dir resolved relative to HF_HOME, matching
    # the cache convention used by other video tasks in the repo.
    override = os.environ.get("MVP_VIDEO_DIR")
    if override:
        return override
    video_cache_dir = config.get("metadata", {}).get("video_cache_dir", "mvp_mini")
    base_cache_dir = os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface/"))
    return os.path.join(base_cache_dir, video_cache_dir)


def mvp_doc_to_visual(doc: Dict[str, Any]) -> List[str]:
    path = os.path.join(_video_root(), doc["video_path"].lstrip("/"))
    if not os.path.exists(path):
        eval_logger.warning(f"MVP video not found: {path} (source={doc.get('source')}). {_SETUP_HINT}")
    return [path]


def _get_candidates(doc: Dict[str, Any]) -> List[str]:
    cands = doc["candidates"]
    if isinstance(cands, str):
        cands = ast.literal_eval(cands)
    return cands


def mvp_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Dict[str, Any] | None = None) -> str:
    kwargs = lmms_eval_specific_kwargs or {}
    post_prompt = kwargs.get("post_prompt", "")
    cands = _get_candidates(doc)
    option_prompt = f"(A) {cands[0]}\n(B) {cands[1]}"
    return f"Question: {doc['question']}\nOption:\n{option_prompt}{post_prompt}"


def mvp_doc_to_answer(doc: Dict[str, Any]) -> str:
    return str(doc["answer"])


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _extract_pred(text: str) -> Union[str, bool]:
    lowered = text.lower()
    pattern = r"(answer|assistant)?:?\s*([ab])\b"
    matches = re.findall(pattern, lowered, re.IGNORECASE)
    if matches:
        chosen = matches[1] if len(matches) > 1 else matches[0]
        return str(IDX_MAP.index(chosen[1].lower()))
    if len(lowered) == 1 and lowered in IDX_MAP:
        return str(IDX_MAP.index(lowered))
    return False


def mvp_process_results(doc: Dict[str, Any], result: List[str]) -> Dict[str, Any]:
    pred = result[0] if result else ""
    answer_pred = _extract_pred(pred)
    cands = _get_candidates(doc)
    answer_idx = str(cands.index(str(doc["answer"])))
    rating = 1 if (answer_pred and answer_pred == answer_idx) else 0

    item = {
        "video_id": doc["video_id"],
        "prediction_idx": answer_pred,
        "answer_idx": answer_idx,
        "rating": rating,
    }
    return {"pair_accuracy": item, "single_accuracy": item}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _compute_pair_and_single(results: List[Dict[str, Any]]):
    """Pair items by video_id stem (drop trailing _<idx>) and compute both
    single-clip accuracy and pair-level accuracy (both clips correct)."""
    single_correct = sum(1 for r in results if r["rating"] == 1)

    by_pair: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        stem = "_".join(r["video_id"].split("_")[:-1])
        by_pair.setdefault(stem, []).append(r)

    pair_correct = 0
    for stem, items in by_pair.items():
        if len(items) == 2 and all(it["rating"] == 1 for it in items):
            pair_correct += 1
        elif len(items) != 2:
            eval_logger.warning("mvp: pair {} has {} items, expected 2", stem, len(items))

    single_acc = single_correct / len(results) * 100 if results else 0.0
    pair_acc = pair_correct / len(by_pair) * 100 if by_pair else 0.0
    return single_acc, pair_acc


def mvp_single_accuracy_aggregation(results: List[Dict[str, Any]]) -> float:
    sa, _ = _compute_pair_and_single(results)
    return sa


def mvp_pair_accuracy_aggregation(results: List[Dict[str, Any]]) -> float:
    _, pa = _compute_pair_and_single(results)
    return pa
