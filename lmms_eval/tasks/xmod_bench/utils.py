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

Media paths are relative to XMODBENCH, e.g. "./benchmark/Data/vggss_audio_bench/foo.wav".
Set the XMODBENCH env var if your data lives somewhere other than the default.
"""

import os
from collections import defaultdict
from typing import Any

import numpy as np
import soundfile as sf
from huggingface_hub import snapshot_download
from loguru import logger as eval_logger
from PIL import Image

HF_REPO_ID = "RyanWW/XModBench"


def _resolve_xmodbench_root() -> str:
    """Return the directory that contains `Data/` and `tasks/`.

    Priority:
      1. $XMODBENCH (must point at a dir laid out like the HF repo: `Data/`, `tasks/` at root).
      2. snapshot_download from RyanWW/XModBench (cached under ~/.cache/huggingface/).
    """
    env = os.getenv("XMODBENCH")
    if env:
        return env
    return snapshot_download(repo_id=HF_REPO_ID, repo_type="dataset")


XMODBENCH_ROOT = _resolve_xmodbench_root()

LETTERS = ["A", "B", "C", "D"]

# Map path prefix → group name (category looks like "01_perception/instruments")
_PATH_PREFIX_TO_GROUP = {
    "01_perception": "perception",
    "02_spatial":    "spatial",
    "03_speech":     "speech",
    "04_temporal":   "temporal",
    "05_exteral":    "external",   # typo in original data
    "05_external":   "external",
}

# VGGSound finegrained plain-English category names → perception/finegrained
_VGGSOUND_CATEGORIES = {
    "animal sounds", "human activities", "human speech",
    "musical instruments", "natural sounds",
    "tools & machinery", "transportation", "urban sounds",
}

_GROUP_ORDER = ["perception", "spatial", "speech", "temporal", "external"]


def _get_group_and_subtask(category: str) -> tuple[str, str]:
    """Return (group, subtask) for a given category string.

    Handles two formats:
    - Path-style  : "01_perception/instruments"  → ("perception", "instruments")
    - Plain English: "Animal Sounds"              → ("perception", "finegrained")
    """
    if "/" in category:
        prefix, subtask = category.split("/", 1)
        group = _PATH_PREFIX_TO_GROUP.get(prefix.lower(), "other")
        return group, subtask

    cat_lower = category.lower()
    if cat_lower in _VGGSOUND_CATEGORIES:
        return "perception", "finegrained"
    if cat_lower == "general_activities":
        return "perception", "general_activities"

    return "other", category


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def resolve_path(relative_path: str) -> str:
    """Convert a dataset-relative path (e.g. `Data/...`) to an absolute path."""
    if os.path.isabs(relative_path):
        return relative_path
    return os.path.join(XMODBENCH_ROOT, relative_path.lstrip("./"))


# ---------------------------------------------------------------------------
# Media loading helpers
# ---------------------------------------------------------------------------


def _load_media(input_path: str, modality: str):
    """Load a media file and return the Python object expected by generate_until.

    - image → PIL.Image.Image
    - audio → {"array": np.ndarray (float32, mono), "sampling_rate": int}
    - video → str (absolute path; the model handles decoding)
    """
    abs_path = resolve_path(input_path)
    mod = modality.lower()
    if mod == "image":
        return Image.open(abs_path).convert("RGB")
    elif mod == "audio":
        array, sr = sf.read(abs_path, dtype="float32")
        if array.ndim > 1:          # stereo → mono
            array = array.mean(axis=1)
        return {"array": array, "sampling_rate": sr}
    elif mod == "video":
        return abs_path             # model handles video decoding
    else:
        raise ValueError(f"Unknown modality: {modality}")


# ---------------------------------------------------------------------------
# doc_to_visual  (legacy fallback; doc_to_messages is preferred)
# ---------------------------------------------------------------------------


def xmod_bench_doc_to_visual(doc: dict) -> list:
    """Return an ordered list of all non-text media as Python objects.

    Order: [condition, optA, optB, optC, optD]  (text items are skipped).
    - image  → PIL.Image.Image
    - audio  → {"array": np.ndarray, "sampling_rate": int}
    - video  → str path
    """
    result = []

    cond = doc["conditions"]
    if cond["modality"].lower() != "text":
        result.append(_load_media(cond["input"], cond["modality"]))

    for letter in LETTERS:
        opt = doc["options"][letter]
        if opt["modality"].lower() != "text":
            result.append(_load_media(opt["input"], opt["modality"]))

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
    def _norm(mod: str) -> str:
        m = mod.lower()
        return "vision" if m in ("image", "video") else m

    cond_mod = _norm(doc["conditions"]["modality"])
    opt_mod  = _norm(list(doc["options"].values())[0]["modality"])
    modality_combo = f"{cond_mod}->{opt_mod}"
    category = doc.get("category") or modality_combo
    group, subtask = _get_group_and_subtask(category)

    return {"xmod_bench_score": {
        "category": category,
        "group":    group,
        "subtask":  subtask,
        "modality": modality_combo,
        "correct":  correct,
    }}


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

    grp_correct:     dict[str, int] = defaultdict(int)
    grp_total:       dict[str, int] = defaultdict(int)
    subtask_correct: dict[str, int] = defaultdict(int)  # keyed "group/subtask"
    subtask_total:   dict[str, int] = defaultdict(int)
    mod_correct:     dict[str, int] = defaultdict(int)
    mod_total:       dict[str, int] = defaultdict(int)

    for r in results:
        grp = r["group"]
        sub = r["subtask"]
        mod = r["modality"]
        key = f"{grp}/{sub}"
        grp_correct[grp]         += r["correct"]
        grp_total[grp]           += 1
        subtask_correct[key]     += r["correct"]
        subtask_total[key]       += 1
        mod_correct[mod]         += r["correct"]
        mod_total[mod]           += 1

    eval_logger.info("=" * 60)
    eval_logger.info(f"XModBench Overall Accuracy: {overall_acc:.2f}%  ({total_correct}/{total})")

    eval_logger.info("\nPer group accuracy:")
    for grp in _GROUP_ORDER + sorted(g for g in grp_total if g not in _GROUP_ORDER):
        if grp not in grp_total:
            continue
        acc = grp_correct[grp] / grp_total[grp] * 100.0
        eval_logger.info(f"  {grp:15s}: {acc:6.2f}%  ({grp_correct[grp]}/{grp_total[grp]})")

    eval_logger.info("\nPer subtask accuracy:")
    for grp in _GROUP_ORDER + sorted(g for g in grp_total if g not in _GROUP_ORDER):
        for key in sorted(k for k in subtask_total if k.startswith(grp + "/")):
            acc = subtask_correct[key] / subtask_total[key] * 100.0
            eval_logger.info(f"  {key:40s}: {acc:6.2f}%  ({subtask_correct[key]}/{subtask_total[key]})")

    eval_logger.info("\nPer modality-combo accuracy:")
    for mod in sorted(mod_total):
        acc = mod_correct[mod] / mod_total[mod] * 100.0
        eval_logger.info(f"  {mod:30s}: {acc:6.2f}%  ({mod_correct[mod]}/{mod_total[mod]})")

    eval_logger.info("=" * 60)
    return round(overall_acc, 5)
