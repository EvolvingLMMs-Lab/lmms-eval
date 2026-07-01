"""OCR macro/micro F1 metric for CC-OCR multi_scene_ocr and multi_lan_ocr tracks.

Directly ported from the official CC-OCR ocr_evaluator.py.
"""

import re
from collections import Counter
from typing import Dict, List, Tuple

# Tracks where tokenization must be character-level (rather than word-level).
_CHAR_LEVEL_SPLITS = {"Arabic", "Japanese", "Korean"}


def _is_char_level(track: str, subset: str) -> bool:
    """Whether to split into characters instead of words."""
    if subset in _CHAR_LEVEL_SPLITS:
        return True
    if subset.startswith("zh"):
        return True
    return False


def _token_normalize(token: str, is_lower: bool, is_alphanum_only: bool) -> str:
    if is_lower:
        token = token.lower()
    if is_alphanum_only:
        token = re.sub(r"[^A-Za-z0-9]+", "", token)
    return token


def _text_normalize_and_tokenize(text: str, is_keep_blank: bool, is_lower: bool, is_alphanum_only: bool) -> List[str]:
    text = text.replace("\t", " ").replace("\n", " ").replace("###", "").replace("***", "")
    text = re.sub(r"\s+", " ", text)
    if not is_keep_blank:
        text = text.replace(" ", "")
    tokens = text.split(" ") if is_keep_blank else list(text)
    tokens = [_token_normalize(t, is_lower, is_alphanum_only) for t in tokens]
    return [t for t in tokens if len(t) > 0]


def _match_count(gt_tokens: List[str], pd_tokens: List[str]) -> int:
    gt_counter = dict(Counter(gt_tokens))
    pd_counter = dict(Counter(pd_tokens))
    right = 0
    for tok, count in gt_counter.items():
        right += min(count, pd_counter.get(tok, 0))
    return right


def _compute_pr_f1(gt_tokens: List[str], pd_tokens: List[str]) -> Tuple[float, float, float, int]:
    right = _match_count(gt_tokens, pd_tokens)
    recall = right / (len(gt_tokens) + 1e-9)
    precision = right / (len(pd_tokens) + 1e-9)
    f1 = 2 * recall * precision / (recall + precision + 1e-9)
    return recall, precision, f1, right


def _tokenize_for_sample(track: str, subset: str, gt: str, pred: str):
    is_word_level = not _is_char_level(track, subset)
    is_lower = True
    is_alphanum_only = track == "multi_scene_ocr" and is_word_level
    gt_tokens = _text_normalize_and_tokenize(str(gt).strip(), is_word_level, is_lower, is_alphanum_only)
    pd_tokens = _text_normalize_and_tokenize(str(pred).strip(), is_word_level, is_lower, is_alphanum_only)
    return gt_tokens, pd_tokens


def compute_track(results: List[Dict], key: str) -> float:
    """Return the aggregated metric value (0..1) for the whole track.

    For each subset (defined by ``result["subset"]``):
      - macro F1: mean of per-sample F1
      - micro F1: total_right / (total_gt + total_pred) style harmonic mean
    Track score = arithmetic mean over subsets (matches official summary).
    """
    if not results:
        return 0.0

    by_subset: Dict[str, List[Dict]] = {}
    for r in results:
        by_subset.setdefault(r["subset"], []).append(r)

    subset_scores: Dict[str, float] = {}
    for subset, samples in by_subset.items():
        macro_f1_list = []
        total_right = total_gt = total_pred = 0
        for s in samples:
            gt_tokens, pd_tokens = _tokenize_for_sample(s["track"], subset, s["gt"], s["pred"])
            recall, precision, f1, right = _compute_pr_f1(gt_tokens, pd_tokens)
            macro_f1_list.append(f1)
            total_right += right
            total_gt += len(gt_tokens)
            total_pred += len(pd_tokens)
        if key == "macro_f1":
            subset_scores[subset] = sum(macro_f1_list) / (len(macro_f1_list) + 1e-9)
        elif key == "micro_f1":
            micro_r = total_right / (total_gt + 1e-9)
            micro_p = total_right / (total_pred + 1e-9)
            subset_scores[subset] = 2 * micro_r * micro_p / (micro_r + micro_p + 1e-9)
        else:
            raise ValueError(f"unknown OCR metric key: {key}")

    return sum(subset_scores.values()) / (len(subset_scores) + 1e-9)


def compute_track_with_breakdown(results: List[Dict]) -> Dict:
    """Compute both macro and micro F1 for every subset (used for logging / submissions)."""
    if not results:
        return {"subsets": {}, "macro_f1": 0.0, "micro_f1": 0.0}

    by_subset: Dict[str, List[Dict]] = {}
    for r in results:
        by_subset.setdefault(r["subset"], []).append(r)

    subset_info: Dict[str, Dict[str, float]] = {}
    for subset, samples in by_subset.items():
        macro_f1_list = []
        total_right = total_gt = total_pred = 0
        for s in samples:
            gt_tokens, pd_tokens = _tokenize_for_sample(s["track"], subset, s["gt"], s["pred"])
            recall, precision, f1, right = _compute_pr_f1(gt_tokens, pd_tokens)
            macro_f1_list.append(f1)
            total_right += right
            total_gt += len(gt_tokens)
            total_pred += len(pd_tokens)
        macro_f1 = sum(macro_f1_list) / (len(macro_f1_list) + 1e-9)
        micro_r = total_right / (total_gt + 1e-9)
        micro_p = total_right / (total_pred + 1e-9)
        micro_f1 = 2 * micro_r * micro_p / (micro_r + micro_p + 1e-9)
        subset_info[subset] = {
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "num": len(samples),
        }
    macro_avg = sum(v["macro_f1"] for v in subset_info.values()) / (len(subset_info) + 1e-9)
    micro_avg = sum(v["micro_f1"] for v in subset_info.values()) / (len(subset_info) + 1e-9)
    return {"subsets": subset_info, "macro_f1": macro_avg, "micro_f1": micro_avg}
