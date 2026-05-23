"""Shared evaluation utilities for MedEvalKit tasks.

Mirrors the answer-extraction and judgement logic from
https://github.com/ECNU-CILAB/MedEvalKit so that scores are comparable.
"""

import fcntl
import os
import re
import zipfile
from difflib import SequenceMatcher
from typing import Dict, List, Optional

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge import Rouge


def extract_zip_once(zip_path: str, dest_dir: str, sentinel_name: str = ".extracted") -> None:
    """Extract ``zip_path`` into ``dest_dir`` safely under concurrent DP workers.

    Multiple DP processes call into the same `_get_*_dir()` initialiser and
    previously raced on ``zipfile.extractall`` — one worker would start writing
    while another read half-extracted files (FileNotFoundError) or both wrote
    over each other (corruption / hangs on large archives like PMC-VQA).

    Fix: an exclusive ``fcntl`` lock around extraction plus a sentinel file that
    is only created after a clean extract, so subsequent callers (and reruns)
    short-circuit.
    """
    sentinel = os.path.join(dest_dir, sentinel_name)
    if os.path.exists(sentinel):
        return
    os.makedirs(dest_dir, exist_ok=True)
    lock_path = os.path.join(dest_dir, ".extract.lock")
    with open(lock_path, "w") as lock_fp:
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
        # Re-check under the lock — another worker may have finished while we waited.
        if os.path.exists(sentinel):
            return
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
        # Atomic completion marker — only written after extractall returns cleanly.
        open(sentinel, "w").close()

# ---------------------------------------------------------------------------
# Answer extraction — matches MedEvalKit parse_response logic
# ---------------------------------------------------------------------------

_ANSWER_PATTERNS = [
    "**answer**:",
    "**answer**",
    "*answer*:",
    "**answer:**",
    "answer is",
    "answer:",
    "final answer is",
    "final answer",
]


def extract_boxed(text: str) -> Optional[str]:
    m = re.search(r"\\boxed\{(.+?)\}", text)
    return m.group(1).strip() if m else None


def extract_tag(text: str, tag: str) -> Optional[str]:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def parse_response(response: str) -> str:
    """Extract the core answer from a verbose model response.

    Mirrors MedEvalKit ``parse_response``: first try boxed / XML-tag
    extraction, then strip common preamble phrases.
    """
    resp = response.strip()

    boxed = extract_boxed(resp)
    if boxed is not None:
        return boxed

    tagged = extract_tag(resp, "answer")
    if tagged is not None:
        return tagged

    resp_lower = resp.lower()
    for pattern in _ANSWER_PATTERNS:
        if pattern in resp_lower:
            idx = resp_lower.index(pattern) + len(pattern)
            resp = resp[idx:]
            resp_lower = resp.lower()

    # Strip leading/trailing punctuation and whitespace left over
    return resp.strip().lstrip(":").strip()


# ---------------------------------------------------------------------------
# Judgement helpers — open / closed VQA
# ---------------------------------------------------------------------------


def judge_yesno(answer: str, response: str) -> bool:
    """XOR logic: match only when exactly one of yes/no is present."""
    resp = response.lower().replace("\n", "").replace(".", "")
    has_yes = "yes" in resp
    has_no = "no" in resp
    if has_yes ^ has_no:
        return answer.lower().strip() in resp
    return False


def judge_close_end(answer: str, response: str) -> bool:
    """Exact-match after stripping newlines and periods."""
    cleaned = response.replace("\n", "").replace(".", "").strip().lower()
    return cleaned == answer.strip().lower()


def tokenize(text: str) -> List[str]:
    return text.lower().replace(".", " .").split()


def bleu(pred: str, target: str, n: int) -> float:
    weights = tuple(1.0 / n for _ in range(n))
    smooth = SmoothingFunction().method1
    return sentence_bleu(
        [tokenize(target)],
        tokenize(pred),
        weights=weights,
        smoothing_function=smooth,
    )


def rouge_scores(pred: str, target: str) -> Dict[str, float]:
    if not pred.strip() or not target.strip():
        return {
            "rouge-1": {"f": 0.0},
            "rouge-2": {"f": 0.0},
            "rouge-l": {"f": 0.0},
        }
    scorer = Rouge()
    try:
        scores = scorer.get_scores(pred.lower(), target.lower())[0]
    except Exception:
        return {
            "rouge-1": {"f": 0.0},
            "rouge-2": {"f": 0.0},
            "rouge-l": {"f": 0.0},
        }
    return scores


def judge_open(answer: str, response: str) -> Dict[str, float]:
    """Compute open-ended VQA metrics."""
    em = float(response.strip().lower() == answer.strip().lower())
    b1 = bleu(response, answer, 1)
    b2 = bleu(response, answer, 2)
    b3 = bleu(response, answer, 3)
    b4 = bleu(response, answer, 4)
    rouge = rouge_scores(response, answer)
    pred_toks = set(tokenize(response))
    gt_toks = set(tokenize(answer))
    common = pred_toks & gt_toks
    if common:
        precision = len(common) / len(pred_toks)
        recall = len(common) / len(gt_toks)
        f1 = 2 * precision * recall / (precision + recall)
    else:
        precision = recall = f1 = 0.0
    return {
        "em": em,
        "bleu1": b1,
        "bleu2": b2,
        "bleu3": b3,
        "bleu4": b4,
        "rouge1": rouge["rouge-1"]["f"],
        "rouge2": rouge["rouge-2"]["f"],
        "rougel": rouge["rouge-l"]["f"],
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ---------------------------------------------------------------------------
# Multi-choice judgement — mirrors MedEvalKit judge_multi_choice
# ---------------------------------------------------------------------------


def judge_multi_choice(choices: List[str], answer: str, response: str) -> bool:
    """Determine if *response* matches *answer* given *choices*."""
    response = response.lower()
    alphas = [chr(ord("a") + i) for i in range(len(choices))]

    # Check first / last paragraph for a bare letter
    first_para = response.split("\n\n")[0]
    if first_para in alphas:
        response = first_para
    else:
        last_part = response.split("\n\n")[-1].split(".")[0]
        if last_part in alphas:
            response = last_part

    response = parse_response(response)
    choices_lower = [c.lower() for c in choices]
    response = response.strip().lower().replace("\n", "")
    split_response = response.split(".")[0]
    split_response = split_response.split(":")[-1].strip()
    answer = answer.strip().lower()

    if len(split_response) > 300:
        pass
    elif split_response == answer:
        return True
    elif split_response in alphas:
        if choices_lower[ord(split_response) - ord("a")] == answer:
            return True
    elif split_response in choices_lower:
        if answer in alphas:
            if split_response == choices_lower[ord(answer) - ord("a")]:
                return True
    else:
        # Fallback: similarity matching
        best_idx = 0
        best_score = 0.0
        for i, choice in enumerate(choices_lower):
            score = SequenceMatcher(None, choice, response).ratio()
            if score > best_score:
                best_score = score
                best_idx = i
        if alphas[best_idx] == answer or choices_lower[best_idx] == answer:
            return True

    return False


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def strip_thinking(text: str) -> str:
    """Remove ``<think>...</think>`` blocks from reasoning models."""
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return stripped if stripped else text


def agg_mean(items: List) -> float:
    valid = [x for x in items if x is not None]
    return sum(valid) / len(valid) if valid else 0.0


def no_image_doc_to_visual(doc) -> list:
    """Return None — no image passed to the model.

    Used for no-image baseline evaluation to test how much of a VQA
    benchmark can be solved from the question text alone.
    """
    return [None]
