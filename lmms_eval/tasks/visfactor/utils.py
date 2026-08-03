"""lmms-eval task adapter for VisFactor.

Paper:  VisFactor: Benchmarking Multimodal Large Language Models on Human
        Visual Cognition (CUHK-ARISE).
Source: https://github.com/CUHK-ARISE/VisFactor
        (data file: VisFactor.tsv from opencompass.openxlab.space)

This module ports the official VLMEvalKit evaluator
(``vlmeval/dataset/visfactor.py``) to the lmms-eval framework.

Scoring (verbatim from the official implementation):

* A *test item* is identified by ``(category_id, eval_index)``; it may span
  multiple rows. An item is correct iff every row is correct (logical AND).
* Per-category accuracy = ``#correct_items / #items``.
* The final score is the macro-average over all categories.
"""

from __future__ import annotations

import io
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from PIL import Image

# ---------------------------------------------------------------------------
# Tokens used in the prompt template.
# ---------------------------------------------------------------------------
_ADDITIONAL_RE = re.compile(r"<ADDITIONAL_(\d+)>")
_IMAGE_RE = re.compile(r"<IMAGE_\d+>")
_JSON_ANSWER_RE = re.compile(r'\{\s*"answer"\s*:\s*(.+?)\s*\}')

# ---------------------------------------------------------------------------
# Category groups (sets are tiny -> use frozenset for clarity & immutability).
# Mirrors the conditional in the official ``evaluate``.
# ---------------------------------------------------------------------------
# Boolean answer in {"T", "F"}.
_BOOL_CIDS = frozenset({"CF1", "CF2", "MV1", "MV2", "MV3", "P3", "RL2", "S1", "S2", "SS2", "VZ1", "VZ2"})
# Answer is a comma-separated list; prediction matches if it's in the list.
_STR_LIST_CIDS = frozenset({"CS1", "CS2", "CS3"})
# Number / coordinate / letter answers.
_NUM_LIKE_CIDS = frozenset({"CF3", "I3", "MA1", "SS3", "VZ3"})

_BOOL_TRUE = frozenset({"t", "y", "1", "true", "yes"})
_BOOL_FALSE = frozenset({"f", "n", "0", "false", "no"})


# ---------------------------------------------------------------------------
# doc_to_visual
# ---------------------------------------------------------------------------
def _to_pil(item: Any) -> Image.Image:
    if isinstance(item, Image.Image):
        return item.convert("RGB")
    if isinstance(item, dict) and item.get("bytes") is not None:
        return Image.open(io.BytesIO(item["bytes"])).convert("RGB")
    raise TypeError(f"unsupported image cell type: {type(item).__name__}")


def visfactor_doc_to_visual(doc: Dict[str, Any]) -> List[Image.Image]:
    return [_to_pil(x) for x in doc["image"]]


# ---------------------------------------------------------------------------
# doc_to_text
# ---------------------------------------------------------------------------
def _resolve_additional(text: str, additional: str) -> str:
    """Substitute ``<ADDITIONAL_i>`` with the i-th ``;``-separated part."""
    parts = additional.replace("<br>", "\n").split(";")

    def repl(m: re.Match) -> str:
        i = int(m.group(1))
        return parts[i] if 0 <= i < len(parts) else m.group(0)

    return _ADDITIONAL_RE.sub(repl, text)


def visfactor_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Dict[str, str] | None = None) -> str:
    text = doc["question"].replace("<br>", "\n")

    # ``additional`` may be NaN (pandas) or "" (already cleaned).
    additional = doc.get("additional", "")
    additional = "" if additional is None else str(additional)
    if additional and additional.lower() != "nan":
        text = _resolve_additional(text, additional)

    # lmms-eval passes images out-of-band -> strip the placeholders.
    text = _IMAGE_RE.sub("", text)

    if lmms_eval_specific_kwargs:
        text = lmms_eval_specific_kwargs.get("pre_prompt", "") + text + lmms_eval_specific_kwargs.get("post_prompt", "")
    return text


# ---------------------------------------------------------------------------
# Prediction extraction helpers (ported verbatim from official impl).
# ---------------------------------------------------------------------------
def _extract_last_json_answer(s: str) -> str:
    matches = list(_JSON_ANSWER_RE.finditer(s))
    if not matches:
        return ""
    raw = matches[-1].group(1).strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ("'", '"'):
        raw = raw[1:-1]
    return raw


def _extract_last_numbers(s: str) -> List[str]:
    return re.findall(r"\d+", s)


def _extract_last_uppercase_letter(s: str) -> str:
    for ch in reversed(s):
        if ch.isupper():
            return ch
    return ""


# ---------------------------------------------------------------------------
# Per-row scoring (1/0). Mirrors the branch structure of the official impl.
# ---------------------------------------------------------------------------
def _score_row(cid: str, raw_pred: str, gold: str, additional: str) -> int:
    pred = _extract_last_json_answer(raw_pred)

    # Boolean branch (also catches VZ3 when ``additional`` is non-numeric).
    if cid in _BOOL_CIDS or (cid == "VZ3" and not additional.isdigit()):
        lower = pred.lower()
        if lower in _BOOL_TRUE:
            tok = "T"
        elif lower in _BOOL_FALSE:
            tok = "F"
        else:
            tok = ""
        return int(tok == gold)

    if cid in _STR_LIST_CIDS:
        choices = [g.strip().lower() for g in gold.split(",")]
        return int(pred.strip().lower() in choices)

    if cid in _NUM_LIKE_CIDS:
        # Official falls back to the raw prediction if JSON extraction fails,
        # so the number/letter scanners still have something to chew on.
        if pred == "":
            pred = raw_pred
        if cid == "VZ3":  # VZ3 + digit-additional -> single uppercase letter
            pred = _extract_last_uppercase_letter(pred)
        elif cid == "CF3":  # coordinate pair
            nums = _extract_last_numbers(pred)
            pred = "" if len(nums) < 2 else f"({nums[-2]}, {nums[-1]})"
        else:  # last integer
            nums = _extract_last_numbers(pred)
            pred = nums[-1] if nums else ""
        return int(pred == gold)

    # Unknown category: fail closed.
    return 0


# ---------------------------------------------------------------------------
# lmms-eval integration
# ---------------------------------------------------------------------------
def visfactor_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    pred = results[0] if results else ""
    additional = doc.get("additional", "") or ""
    correct = _score_row(
        cid=str(doc["category_id"]),
        raw_pred=str(pred),
        gold=str(doc["answer"]),
        additional=str(additional),
    )
    return {
        "visfactor_score": {
            "category_id": str(doc["category_id"]),
            "eval_index": int(doc["eval_index"]),
            "correct": int(correct),
        }
    }


def visfactor_aggregate(results: List[Dict[str, Any]]) -> float:
    """AND over rows of an item -> mean over items -> macro over categories."""
    # (category_id, eval_index) -> list of per-row correctness
    items: Dict[Tuple[str, int], List[int]] = defaultdict(list)
    for r in results:
        items[(r["category_id"], r["eval_index"])].append(r["correct"])

    # Collapse rows with AND, then group by category.
    per_cat: Dict[str, List[int]] = defaultdict(list)
    for (cid, _eid), rows in items.items():
        per_cat[cid].append(int(all(rows)))

    if not per_cat:
        return 0.0
    cat_acc = [sum(v) / len(v) for v in per_cat.values()]
    return sum(cat_acc) / len(cat_acc)
