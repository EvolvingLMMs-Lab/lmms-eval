"""
XModBench utilities for lmms-eval.

Dataset schema (per sample):
  {
    "question": str,
    "conditions": {"modality": "Audio"|"Image"|"Video"|"Text", "input": str},
    "options": {
      "A": {"modality": ..., "input": str},
      "B": {"modality": ..., "input": str},
      "C": {"modality": ..., "input": str},
      "D": {"modality": ..., "input": str},
    },
    "correct_answer": "A"|"B"|"C"|"D",
    "category": str  # optional
  }

Media paths are relative to AUDIOBENCH_ROOT, e.g. "./benchmark/Data/vggss_audio_bench/foo.wav".
Set the AUDIOBENCH_ROOT env var if your data lives somewhere other than the default.
"""

import os
from collections import defaultdict
from typing import Any

import numpy as np
from loguru import logger as eval_logger

AUDIOBENCH_ROOT = os.getenv("AUDIOBENCH_ROOT", "/home/xwang378/scratch/2025/AudioBench")

LETTERS = ["A", "B", "C", "D"]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def resolve_path(relative_path: str) -> str:
    """Convert a dataset-relative path (./benchmark/...) to an absolute path."""
    if os.path.isabs(relative_path):
        return relative_path
    # Strip leading "./"
    if relative_path.startswith("./"):
        relative_path = relative_path[2:]
    return os.path.join(AUDIOBENCH_ROOT, relative_path)


# ---------------------------------------------------------------------------
# doc_to_visual  (legacy fallback; doc_to_messages is preferred)
# ---------------------------------------------------------------------------


def xmod_bench_doc_to_visual(doc: dict) -> list:
    """Return an ordered list of all non-text media paths.

    Order: [condition, optA, optB, optC, optD]  (text items are skipped).
    """
    result = []

    cond = doc["conditions"]
    if cond["modality"].lower() != "text":
        result.append(resolve_path(cond["input"]))

    for letter in LETTERS:
        opt = doc["options"][letter]
        if opt["modality"].lower() != "text":
            result.append(resolve_path(opt["input"]))

    return result


# ---------------------------------------------------------------------------
# doc_to_messages  (recommended: interleaved multi-modal messages)
# ---------------------------------------------------------------------------


def xmod_bench_doc_to_messages(doc: dict, lmms_eval_specific_kwargs: dict | None = None) -> list:
    """Build a single-turn user message with interleaved media + text.

    Layout:
        [condition_media?]
        <question text>
        A: [optA_media | optA_text]
        B: [optB_media | optB_text]
        C: [optC_media | optC_text]
        D: [optD_media | optD_text]
        <post_prompt>
    """
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    post_prompt = lmms_eval_specific_kwargs.get(
        "post_prompt", "Answer with the option's letter (A, B, C, or D) directly."
    )

    content: list[dict[str, Any]] = []

    # -- Condition ----------------------------------------------------------
    cond = doc["conditions"]
    cond_mod = cond["modality"].lower()
    if cond_mod == "text":
        # Prepend text condition inline with the question
        content.append({"type": "text", "text": f"Context: {cond['input']}\n\n"})
    else:
        content.append({"type": cond_mod, "url": resolve_path(cond["input"])})

    # -- Question -----------------------------------------------------------
    content.append({"type": "text", "text": doc["question"]})

    # -- Options ------------------------------------------------------------
    for letter in LETTERS:
        opt = doc["options"][letter]
        opt_mod = opt["modality"].lower()
        if opt_mod == "text":
            content.append({"type": "text", "text": f"\n{letter}. {opt['input']}"})
        else:
            content.append({"type": "text", "text": f"\n{letter}: "})
            content.append({"type": opt_mod, "url": resolve_path(opt["input"])})

    # -- Post-prompt --------------------------------------------------------
    content.append({"type": "text", "text": f"\n{post_prompt}"})

    return [{"role": "user", "content": content}]


# ---------------------------------------------------------------------------
# doc_to_text  (legacy text-only fallback, media replaced by placeholders)
# ---------------------------------------------------------------------------


def xmod_bench_doc_to_text(doc: dict, lmms_eval_specific_kwargs: dict | None = None) -> str:
    """Format the prompt as plain text (media items become <media_N> tokens)."""
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    post_prompt = lmms_eval_specific_kwargs.get(
        "post_prompt", "Answer with the option's letter (A, B, C, or D) directly."
    )

    media_idx = 0
    parts = []

    cond = doc["conditions"]
    cond_mod = cond["modality"].lower()
    if cond_mod == "text":
        parts.append(f"Context: {cond['input']}\n\n")
    else:
        parts.append(f"<media_{media_idx}>\n")
        media_idx += 1

    parts.append(doc["question"])

    for letter in LETTERS:
        opt = doc["options"][letter]
        opt_mod = opt["modality"].lower()
        if opt_mod == "text":
            parts.append(f"\n{letter}. {opt['input']}")
        else:
            parts.append(f"\n{letter}: <media_{media_idx}>")
            media_idx += 1

    parts.append(f"\n{post_prompt}")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_multi_choice_response(response: str, all_choices: list[str]) -> str | None:
    """Extract a letter answer (A/B/C/D) from a free-form model response."""
    response = " " + response.strip() + " "

    # 1. Try "(A)" style
    candidates = [c for c in all_choices if f"({c})" in response]

    # 2. Try "A " style
    if not candidates:
        candidates = [c for c in all_choices if f" {c} " in response]

    # 3. Try "A." style
    if not candidates:
        candidates = [c for c in all_choices if f"{c}." in response]

    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0]

    # Multiple candidates: return the last occurring one
    start_indexes = [response.rfind(f" {c} ") for c in candidates]
    return candidates[int(np.argmax(start_indexes))]


# ---------------------------------------------------------------------------
# process_results
# ---------------------------------------------------------------------------


def xmod_bench_process_results(doc: dict, results: list) -> dict:
    """Parse model prediction and compare to ground truth.

    Returns a dict suitable for aggregation:
        {"xmod_bench_score": {"category": str, "correct": 0|1}}
    """
    response = results[0].strip() if results else ""
    pred = parse_multi_choice_response(response, LETTERS)

    gold = doc["correct_answer"].strip().upper()
    correct = int(pred == gold) if pred is not None else 0

    # Derive category: use explicit field, or fall back to modality combo
    cond_mod = doc["conditions"]["modality"].lower()
    opt_mod = list(doc["options"].values())[0]["modality"].lower()
    modality_combo = f"{cond_mod}->{opt_mod}"
    category = doc.get("category") or modality_combo

    return {"xmod_bench_score": {"category": category, "modality": modality_combo, "correct": correct}}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def xmod_bench_aggregate_results(results: list) -> float:
    """Compute overall and per-category accuracy, log a summary, return overall %."""
    total = len(results)
    if total == 0:
        return 0.0

    total_correct = sum(r["correct"] for r in results)
    overall_acc = total_correct / total * 100.0

    # Per-category
    cat_correct: dict[str, int] = defaultdict(int)
    cat_total: dict[str, int] = defaultdict(int)
    # Per-modality-combo
    mod_correct: dict[str, int] = defaultdict(int)
    mod_total: dict[str, int] = defaultdict(int)

    for r in results:
        cat = r["category"]
        mod = r["modality"]
        cat_correct[cat] += r["correct"]
        cat_total[cat] += 1
        mod_correct[mod] += r["correct"]
        mod_total[mod] += 1

    eval_logger.info("=" * 60)
    eval_logger.info(f"XModBench Overall Accuracy: {overall_acc:.2f}%  ({total_correct}/{total})")

    eval_logger.info("\nPer modality-combo accuracy:")
    for mod in sorted(mod_total):
        acc = mod_correct[mod] / mod_total[mod] * 100.0
        eval_logger.info(f"  {mod:30s}: {acc:6.2f}%  ({mod_correct[mod]}/{mod_total[mod]})")

    eval_logger.info("\nPer category accuracy:")
    for cat in sorted(cat_total):
        acc = cat_correct[cat] / cat_total[cat] * 100.0
        eval_logger.info(f"  {cat:40s}: {acc:6.2f}%  ({cat_correct[cat]}/{cat_total[cat]})")

    eval_logger.info("=" * 60)
    return round(overall_acc, 5)
