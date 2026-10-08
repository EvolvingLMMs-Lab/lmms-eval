"""DIVE-Bench objective metrics and frozen public prompt protocol.

This module makes no judge API calls and does not download videos at import.
The scoring kernels below preserve the released per-example metric semantics.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

# CER/WER are always complete; never silently skip long references.
FAST_TEXT_METRICS = False
MAX_EDIT_DISTANCE_CELLS = 25_000_000
HIGHMOTION_ORDERED_CONTENT_SHA256 = "90ee915016105f6a709f391e8a03a6d0e99bc5c908f945cdf7b80d0cb289e789"


def _highmotion_frame_budget():
    value = os.getenv("DENSEVIDEO_HIGHMOTION_NUM_FRAMES", "8").strip()
    try:
        budget = int(value)
    except ValueError as exc:
        raise ValueError(f"DENSEVIDEO_HIGHMOTION_NUM_FRAMES must be an integer, got {value!r}") from exc
    if budget <= 0:
        raise ValueError(f"DENSEVIDEO_HIGHMOTION_NUM_FRAMES must be positive, got {budget}")
    return budget


def _uniform_subsample(sequence, sample_count):
    """Match the wrappers' endpoint-inclusive uniform frame sampling."""

    sequence = list(sequence)
    if not sequence:
        return []
    sample_count = min(int(sample_count), len(sequence))
    indices = np.linspace(0, len(sequence) - 1, sample_count, dtype=int)
    return [sequence[int(index)] for index in indices]


def _normalize_text(text):
    if text is None:
        return ""
    text = str(text).strip().lower()
    return " ".join(text.split())


def _levenshtein_distance(seq_a, seq_b):
    len_a = len(seq_a)
    len_b = len(seq_b)
    if len_a == 0:
        return len_b
    if len_b == 0:
        return len_a

    prev = list(range(len_b + 1))
    for i in range(1, len_a + 1):
        curr = [i] + [0] * len_b
        for j in range(1, len_b + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[len_b]


def _compute_cer(pred, ref):
    pred_chars = list(_normalize_text(pred))
    ref_chars = list(_normalize_text(ref))
    if FAST_TEXT_METRICS and len(pred_chars) * len(ref_chars) > MAX_EDIT_DISTANCE_CELLS:
        return float("nan")
    return _levenshtein_distance(pred_chars, ref_chars) / max(len(ref_chars), 1)


def _compute_wer(pred, ref):
    pred_words = _normalize_text(pred).split()
    ref_words = _normalize_text(ref).split()
    if FAST_TEXT_METRICS and len(pred_words) * len(ref_words) > MAX_EDIT_DISTANCE_CELLS:
        return float("nan")
    return _levenshtein_distance(pred_words, ref_words) / max(len(ref_words), 1)


def _compute_exact_match(pred, ref):
    return float(_normalize_text(pred) == _normalize_text(ref))


def _compute_token_f1(pred, ref):
    pred_tokens = _normalize_text(pred).split()
    ref_tokens = _normalize_text(ref).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts = {}
    ref_counts = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1

    overlap = 0
    for token, count in pred_counts.items():
        if token in ref_counts:
            overlap += min(count, ref_counts[token])

    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


GRID_LABELS = {
    "r1c1": (1.0 / 6.0, 1.0 / 6.0),
    "r1c2": (3.0 / 6.0, 1.0 / 6.0),
    "r1c3": (5.0 / 6.0, 1.0 / 6.0),
    "r2c1": (1.0 / 6.0, 3.0 / 6.0),
    "r2c2": (3.0 / 6.0, 3.0 / 6.0),
    "r2c3": (5.0 / 6.0, 3.0 / 6.0),
    "r3c1": (1.0 / 6.0, 5.0 / 6.0),
    "r3c2": (3.0 / 6.0, 5.0 / 6.0),
    "r3c3": (5.0 / 6.0, 5.0 / 6.0),
}

GRID_ALIASES = {
    "1": "r1c1",
    "2": "r1c2",
    "3": "r1c3",
    "4": "r2c1",
    "5": "r2c2",
    "6": "r2c3",
    "7": "r3c1",
    "8": "r3c2",
    "9": "r3c3",
    "topleft": "r1c1",
    "top_left": "r1c1",
    "top left": "r1c1",
    "upperleft": "r1c1",
    "upper_left": "r1c1",
    "upper left": "r1c1",
    "lefttop": "r1c1",
    "left_top": "r1c1",
    "top": "r1c2",
    "topcenter": "r1c2",
    "top_center": "r1c2",
    "top center": "r1c2",
    "topmiddle": "r1c2",
    "top_middle": "r1c2",
    "top middle": "r1c2",
    "upper": "r1c2",
    "uppercenter": "r1c2",
    "upper_center": "r1c2",
    "upper center": "r1c2",
    "topright": "r1c3",
    "top_right": "r1c3",
    "top right": "r1c3",
    "upperright": "r1c3",
    "upper_right": "r1c3",
    "upper right": "r1c3",
    "righttop": "r1c3",
    "right_top": "r1c3",
    "left": "r2c1",
    "middleleft": "r2c1",
    "middle_left": "r2c1",
    "middle left": "r2c1",
    "centerleft": "r2c1",
    "center_left": "r2c1",
    "center left": "r2c1",
    "middle": "r2c2",
    "center": "r2c2",
    "centre": "r2c2",
    "r2c2": "r2c2",
    "right": "r2c3",
    "middleright": "r2c3",
    "middle_right": "r2c3",
    "middle right": "r2c3",
    "centerright": "r2c3",
    "center_right": "r2c3",
    "center right": "r2c3",
    "bottomleft": "r3c1",
    "bottom_left": "r3c1",
    "bottom left": "r3c1",
    "lowerleft": "r3c1",
    "lower_left": "r3c1",
    "lower left": "r3c1",
    "leftbottom": "r3c1",
    "left_bottom": "r3c1",
    "bottom": "r3c2",
    "bottomcenter": "r3c2",
    "bottom_center": "r3c2",
    "bottom center": "r3c2",
    "bottommiddle": "r3c2",
    "bottom_middle": "r3c2",
    "bottom middle": "r3c2",
    "lower": "r3c2",
    "lowercenter": "r3c2",
    "lower_center": "r3c2",
    "lower center": "r3c2",
    "bottomright": "r3c3",
    "bottom_right": "r3c3",
    "bottom right": "r3c3",
    "lowerright": "r3c3",
    "lower_right": "r3c3",
    "lower right": "r3c3",
    "rightbottom": "r3c3",
    "right_bottom": "r3c3",
}

_GRID_MAX_DISTANCE = math.sqrt(2.0)


def _canonical_grid_label(label):
    if label is None:
        return None
    text = str(label).strip().lower()
    text = text.strip("`'\".,;:()[]{}")
    text = re.sub(r"[\s\-]+", " ", text)
    compact = text.replace(" ", "")
    text_us = text.replace(" ", "_")
    for candidate in (text, compact, text_us):
        if candidate in GRID_LABELS:
            return candidate
        if candidate in GRID_ALIASES:
            return GRID_ALIASES[candidate]
    match = re.fullmatch(r"r\s*([1-3])\s*c\s*([1-3])", text)
    if match:
        return f"r{match.group(1)}c{match.group(2)}"
    match = re.fullmatch(r"row\s*([1-3])\s*(?:col|column)\s*([1-3])", text)
    if match:
        return f"r{match.group(1)}c{match.group(2)}"
    return None


def parse_grid_sequence(text):
    """Parse 3x3 grid labels from JSON lists, comma text, or natural language."""
    if text is None:
        return []

    if isinstance(text, (list, tuple)):
        out = []
        for item in text:
            nested = parse_grid_sequence(item)
            if nested:
                out.extend(nested)
            else:
                label = _canonical_grid_label(item)
                if label:
                    out.append(label)
        return out

    if isinstance(text, dict):
        for key in ("answer", "trajectory", "traj", "sequence", "grid", "positions", "labels"):
            if key in text:
                parsed = parse_grid_sequence(text[key])
                if parsed:
                    return parsed
        out = []
        for value in text.values():
            out.extend(parse_grid_sequence(value))
        return out

    raw = str(text).strip()
    if not raw:
        return []

    for parser in (json.loads, ast.literal_eval):
        if raw[:1] in "[{\"'(":
            try:
                parsed = parser(raw)
                if parsed is not raw:
                    seq = parse_grid_sequence(parsed)
                    if seq:
                        return seq
            except Exception:
                pass

    lowered = raw.lower()
    lowered = lowered.replace("_", " ").replace("-", " ")

    matches = []

    for match in re.finditer(r"\br\s*([1-3])\s*c\s*([1-3])\b", lowered):
        matches.append((match.start(), match.end(), f"r{match.group(1)}c{match.group(2)}"))
    for match in re.finditer(r"\brow\s*([1-3])\s*(?:col|column)\s*([1-3])\b", lowered):
        matches.append((match.start(), match.end(), f"r{match.group(1)}c{match.group(2)}"))

    alias_items = sorted(
        ((alias.replace("_", " ").replace("-", " "), canonical) for alias, canonical in GRID_ALIASES.items() if not alias.isdigit()),
        key=lambda x: len(x[0]),
        reverse=True,
    )
    for alias, canonical in alias_items:
        pattern = r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])"
        for match in re.finditer(pattern, lowered):
            matches.append((match.start(), match.end(), canonical))

    for match in re.finditer(r"(?<!\d)([1-9])(?!\d)", lowered):
        matches.append((match.start(), match.end(), GRID_ALIASES[match.group(1)]))

    if matches:
        chosen = []
        occupied = []
        for start, end, label in sorted(matches, key=lambda x: (x[0], -(x[1] - x[0]))):
            if any(not (end <= s or start >= e) for s, e in occupied):
                continue
            chosen.append((start, label))
            occupied.append((start, end))
        return [label for _, label in sorted(chosen, key=lambda x: x[0])]

    out = []
    for chunk in re.split(r"[,;/\n]+|\s+then\s+|\s*->\s*", lowered):
        label = _canonical_grid_label(chunk)
        if label:
            out.append(label)
    return out


def grid_label_to_xy(label):
    canonical = _canonical_grid_label(label)
    if canonical is None:
        return None
    return GRID_LABELS.get(canonical)


def _mean_finite(values, default=0.0):
    vals = []
    for value in values:
        try:
            value = float(value)
        except Exception:
            continue
        if math.isfinite(value):
            vals.append(value)
    return float(np.mean(vals)) if vals else default


def _grid_sequence_metrics(pred_seq, ref_seq):
    if not ref_seq:
        return {
            "grid_acc": 0.0,
            "grid_ade": _GRID_MAX_DISTANCE,
            "grid_fde": _GRID_MAX_DISTANCE,
            "grid_transition_acc": 0.0,
        }

    correct = 0
    distances = []
    for idx, ref_label in enumerate(ref_seq):
        pred_label = pred_seq[idx] if idx < len(pred_seq) else None
        if pred_label == ref_label:
            correct += 1
        pred_xy = grid_label_to_xy(pred_label)
        ref_xy = grid_label_to_xy(ref_label)
        if pred_xy is None or ref_xy is None:
            distances.append(_GRID_MAX_DISTANCE)
        else:
            distances.append(math.dist(pred_xy, ref_xy))

    last_pred = pred_seq[len(ref_seq) - 1] if len(pred_seq) >= len(ref_seq) else None
    last_ref = ref_seq[-1]
    last_pred_xy = grid_label_to_xy(last_pred)
    last_ref_xy = grid_label_to_xy(last_ref)
    if last_pred_xy is None or last_ref_xy is None:
        fde = _GRID_MAX_DISTANCE
    else:
        fde = math.dist(last_pred_xy, last_ref_xy)

    if len(ref_seq) <= 1:
        transition_acc = 1.0
    else:
        trans_correct = 0
        for idx in range(len(ref_seq) - 1):
            if idx + 1 >= len(pred_seq):
                continue
            ref_a = grid_label_to_xy(ref_seq[idx])
            ref_b = grid_label_to_xy(ref_seq[idx + 1])
            pred_a = grid_label_to_xy(pred_seq[idx])
            pred_b = grid_label_to_xy(pred_seq[idx + 1])
            if None in (ref_a, ref_b, pred_a, pred_b):
                continue
            ref_delta = (round(ref_b[0] - ref_a[0], 6), round(ref_b[1] - ref_a[1], 6))
            pred_delta = (round(pred_b[0] - pred_a[0], 6), round(pred_b[1] - pred_a[1], 6))
            if pred_delta == ref_delta:
                trans_correct += 1
        transition_acc = trans_correct / float(len(ref_seq) - 1)

    return {
        "grid_acc": correct / float(len(ref_seq)),
        "grid_ade": float(np.mean(distances)) if distances else _GRID_MAX_DISTANCE,
        "grid_fde": float(fde),
        "grid_transition_acc": float(transition_acc),
    }


_HIGHMOTION_GRID_NAMES = {
    "r1c1": "topleft",
    "r1c2": "top",
    "r1c3": "topright",
    "r2c1": "left",
    "r2c2": "middle",
    "r2c3": "right",
    "r3c1": "bottomleft",
    "r3c2": "bottom",
    "r3c3": "bottomright",
}


def _highmotion_sampled_grid_sequence(doc):
    full_sequence = parse_grid_sequence(doc.get("answer", ""))
    return _uniform_subsample(full_sequence, _highmotion_frame_budget())


def _highmotion_count_text(value):
    words = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
        11: "eleven",
        12: "twelve",
    }
    return words.get(int(value), str(value))


def highmotion_doc_to_answer(doc):
    """Return ground truth aligned with the uniformly sampled model frames."""

    sampled = _highmotion_sampled_grid_sequence(doc)
    return ",".join(_HIGHMOTION_GRID_NAMES.get(label, label) for label in sampled)


def highmotion_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Ask only for positions observable in the model's sampled video frames."""

    sampled_count = len(_highmotion_sampled_grid_sequence(doc))
    sampled_count_text = _highmotion_count_text(sampled_count)
    comma_count_text = _highmotion_count_text(max(sampled_count - 1, 0))
    original_question = str(doc.get("question", "")).strip()
    action = re.split(r"\n\nWe consider all\s+\d+\s+frames", original_question, maxsplit=1)[0].strip()
    if not action:
        action = "Track the visible right-hand palm center in the video."
    return (
        f"{action}\n\n"
        f"The video input contains exactly {sampled_count_text} frames uniformly sampled in temporal order from the original clip, including its first and last frames. "
        "On each sampled frame, divide the image into top, middle, and bottom rows and left, center, and right columns, then locate the visible right-hand palm center. "
        "Use topleft, top, or topright for the top row; left, middle, or right for the middle row; and bottomleft, bottom, or bottomright for the bottom row. "
        "For every sampled frame in order, select one region; repeat a name when the hand stays in the same region.\n\n"
        f"Answer with exactly {sampled_count_text} labels in sampled-frame order (exactly {comma_count_text} commas) and no extra text. "
        "Return one item per input frame—not one item per possible region—and stop after the final frame's label. "
        f"Do not stop early: if uncertain, repeat your best region choice until all {sampled_count_text} frame slots are filled. "
        "Before answering, verify the comma count. Use commas only; do not use vertical bars or brackets."
    )


def doc_to_visual(doc: dict[str, Any]) -> list[str]:
    """Resolve videos without collapsing high-motion action directories.

    Set DIVE_BENCH_DATA_ROOT (or legacy DENSEVIDEO_DATA_ROOT) to the directory
    containing DenseVideo-LPM/videos and egodex/<action>/<clip>.mp4.
    An educational archive's videos/<basename> layout is also accepted.
    """
    raw = str(doc["video_path"])
    relative = Path(raw)
    if ".." in relative.parts:
        raise ValueError("DIVE-Bench video paths cannot contain parent traversal")
    if relative.is_absolute():
        if relative.is_file():
            return [str(relative)]
        raise FileNotFoundError(f"Missing DIVE-Bench video: {relative}")
    roots = [os.getenv("DIVE_BENCH_DATA_ROOT"), os.getenv("DENSEVIDEO_DATA_ROOT")]
    hf_root = Path(os.getenv("HF_HOME", "~/.cache/huggingface")).expanduser()
    roots += [str(hf_root), str(hf_root / "DenseVideoEvaluation"), str(hf_root / "highmotion_densevideounderstand" / "all_test")]
    for value in dict.fromkeys(filter(None, roots)):
        root = Path(value).expanduser()
        candidates = [root / relative]
        if relative.parts and relative.parts[0] == "DenseVideo-LPM":
            candidates += [root / "videos" / relative.name, root / "DenseVideoEvaluation" / "videos" / relative.name]
        for path in candidates:
            if path.is_file():
                return [str(path)]
    raise FileNotFoundError(f"Missing DIVE-Bench video {raw!r}; set DIVE_BENCH_DATA_ROOT and preserve the annotation-relative directory layout.")


def educational_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs: dict | None = None) -> str:
    """Keep the published educational question and explicit prompt suffix."""
    options = lmms_eval_specific_kwargs or {}
    return f"{options.get('pre_prompt', '')}{doc['question']}{options.get('post_prompt', '')}"


def educational_doc_to_messages(doc: dict[str, Any], lmms_eval_specific_kwargs: dict | None = None) -> list[dict]:
    """Expose the same question/video to chat-template models."""
    content = [{"type": "video", "url": path} for path in doc_to_visual(doc)]
    content.append({"type": "text", "text": educational_doc_to_text(doc, lmms_eval_specific_kwargs)})
    return [{"role": "user", "content": content}]


def highmotion_doc_to_messages(doc: dict[str, Any], lmms_eval_specific_kwargs: dict | None = None) -> list[dict]:
    """Expose the published high-motion question/video to chat models."""
    content = [{"type": "video", "url": path} for path in doc_to_visual(doc)]
    content.append({"type": "text", "text": highmotion_doc_to_text(doc, lmms_eval_specific_kwargs)})
    return [{"role": "user", "content": content}]


def educational_process_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    """Return per-question objective metrics, with no synthetic judge scores."""
    pred = results[0] if results else ""
    ref = doc["answer"]
    return {"cer": _compute_cer(pred, ref), "wer": _compute_wer(pred, ref), "token_f1": _compute_token_f1(pred, ref), "exact_match": _compute_exact_match(pred, ref)}


def highmotion_process_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    """Score trajectories at the configured uniform sampled-frame positions."""
    ref = parse_grid_sequence(highmotion_doc_to_answer(doc))
    pred = parse_grid_sequence(results[0] if results else "")
    return {**_grid_sequence_metrics(pred, ref), "token_f1": _compute_token_f1(" ".join(pred), " ".join(ref))}


def validate_educational(dataset: Any) -> Any:
    """Fail closed if the pinned 634-question release changes."""
    if len(dataset) != 634:
        raise ValueError(f"Expected 634 educational questions, found {len(dataset)}; load only LPM_videos.parquet.")
    return dataset


def validate_highmotion(dataset: Any) -> Any:
    """Validate the full 3,243-row track; qid alone is not a unique key."""
    if len(dataset) != 3243:
        raise ValueError(f"Expected 3243 high-motion clips, found {len(dataset)}.")
    paths = list(dataset["video_path"])
    if len(set(paths)) != len(paths):
        raise ValueError("High-motion video_path must be unique; preserve action directories.")
    rows = [[str(row["video_path"]), str(row["qid"]), str(row["question"]), str(row["answer"]), int(row["frame_count"])] for row in dataset]
    digest = hashlib.sha256(json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    if digest != HIGHMOTION_ORDERED_CONTENT_SHA256:
        raise ValueError(f"High-motion annotation content/order differs from the audited release: {digest}.")
    return dataset


def highmotion_preview_1000(dataset: Any) -> Any:
    """Select exactly the first 1,000 source rows, independent of environment."""
    validate_highmotion(dataset)
    return dataset.select(range(1000))
