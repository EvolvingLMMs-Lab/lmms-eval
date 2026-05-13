"""
Utility functions for the Multi-Level Existence Benchmark (MLE-Bench).

Dataset: JunlinHan/Multi-Level_Existence_Bench
Paper: "Learning to See Before Seeing" (https://junlinhan.github.io/projects/lsbs/)

Dataset schema:
  - image:       PIL Image (imagefolder, loaded from images/ dir)
  - question:    str  e.g. "Which of the following appears in this image?"
  - choices:     list[str]  e.g. ["curtain", "television", "cabinet", "mirror"]
  - answer:      str  the correct answer TEXT  e.g. "cabinet"
  - answer_text: str  same as answer
  - sub_task:    str  "existence_0-30" | "existence_30-60" | "existence_60-100"

The three sub-tasks correspond to object size (percentage of image pixels):
  - existence_0-30  → small  objects  (732 samples)
  - existence_30-60 → medium objects  (698 samples)
  - existence_60-100→ large  objects  (431 samples)
"""

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

from loguru import logger as eval_logger

# Map sub_task string to a human-readable category name
SUB_TASK_MAP = {
    "existence_0-30": "small",
    "existence_30-60": "medium",
    "existence_60-100": "large",
}


# ---------------------------------------------------------------------------
# Helper: extract answer letter from model output
# ---------------------------------------------------------------------------


def _extract_answer_letter(text: str) -> str:
    """
    Robustly extract the first answer-choice letter (A/B/C/D) from model output.

    Handles common patterns:
      "A"  /  "A."  /  "(A)"  /  "A)"  /  "A answer..."
      "The answer is A"  /  "Option A"
    Returns the letter in upper-case, or "" if nothing found.
    """
    text = text.strip()
    # Explicit "the answer is X" / "answer: X" / "answer is: X"
    m = re.search(r"(?:the\s+answer\s+is\s*[:\-]?|answer\s*[:\-])\s*[\(\[]?([A-Da-d])[\)\]]?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # "Option X" / "option X"
    m = re.search(r"option\s+([A-Da-d])", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Leading letter with optional brackets / punctuation
    m = re.match(r"[\(\[]*([A-Da-d])[\)\]\.:\s]", text)
    if m:
        return m.group(1).upper()
    # Bare single letter
    if len(text) == 1 and text.upper() in "ABCD":
        return text.upper()
    return ""


def _answer_to_letter(doc: Dict) -> str:
    """
    Convert the text answer to a letter (A/B/C/D) based on its position in choices.
    Returns "" if the answer is not found in choices.
    """
    choices = doc["choices"]
    answer = doc["answer"]
    try:
        idx = choices.index(answer)
        return chr(65 + idx)  # 0->A, 1->B, 2->C, 3->D
    except ValueError:
        # Fallback: case-insensitive match
        for i, c in enumerate(choices):
            if c.strip().lower() == answer.strip().lower():
                return chr(65 + i)
        eval_logger.warning(f"Answer '{answer}' not found in choices {choices}")
        return ""


# ---------------------------------------------------------------------------
# Core task functions
# ---------------------------------------------------------------------------


def mle_bench_doc_to_visual(doc: Dict) -> List:
    """Return the image as a list (lmms-eval convention)."""
    return [doc["image"].convert("RGB")]


def mle_bench_doc_to_text(doc: Dict, lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> str:
    """
    Format the question and choices into a text prompt.

    Example output:
        Which of the following appears in this image?
        A. curtain
        B. television
        C. cabinet
        D. mirror
        Answer with the option's letter from the given choices directly
    """
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    question = doc["question"].strip()
    choices = doc["choices"]
    choices_text = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))

    return f"{pre_prompt}{question}\n{choices_text}{post_prompt}"


def mle_bench_doc_to_target(doc: Dict) -> str:
    """
    Return the ground-truth answer letter (A/B/C/D).
    lmms-eval compares model output against this value.
    """
    return _answer_to_letter(doc)


# ---------------------------------------------------------------------------
# Result processing & aggregation
# ---------------------------------------------------------------------------


def mle_bench_process_results(doc: Dict, results: List[str]) -> Dict:
    """
    Parse model output and compute per-sample correctness.

    Returns a dict keyed by sub_task category + "average", each containing
    a list entry that aggregate_results will collect.
    """
    pred_raw = results[0] if results else ""
    pred_letter = _extract_answer_letter(pred_raw)
    gt_letter = _answer_to_letter(doc)

    is_correct = (pred_letter == gt_letter) and (gt_letter != "")

    sub_task = doc.get("sub_task", "unknown")
    category = SUB_TASK_MAP.get(sub_task, sub_task)

    record = {
        "id": doc.get("id", ""),
        "sub_task": sub_task,
        "category": category,
        "gt_letter": gt_letter,
        "pred_letter": pred_letter,
        "pred_raw": pred_raw,
        "is_correct": is_correct,
    }

    return {
        category: record,
        "average": record,
    }


def mle_bench_aggregate_results(results: List[Dict]) -> float:
    """
    Compute accuracy grouped by size category, then report overall average.

    The final returned score is the macro-average across the three categories
    (small / medium / large), which is the primary metric used in the paper.
    """
    category_scores: Dict[str, List[bool]] = defaultdict(list)

    for r in results:
        category_scores[r["category"]].append(r["is_correct"])

    category_acc: Dict[str, float] = {}
    for cat, scores in sorted(category_scores.items()):
        acc = sum(scores) / len(scores) if scores else 0.0
        category_acc[cat] = acc
        eval_logger.info(f"MLE-Bench [{cat:8s}]: {acc * 100:.2f}%  ({sum(scores)}/{len(scores)})")

    if not category_acc:
        return 0.0

    # Macro-average over categories present in this run
    macro_avg = sum(category_acc.values()) / len(category_acc)
    eval_logger.info(f"MLE-Bench [overall ]: {macro_avg * 100:.2f}%")
    return macro_avg


# ---------------------------------------------------------------------------
# Filter functions for sub-task specific YAML configs
# ---------------------------------------------------------------------------


def filter_small(dataset):
    """Keep only small-object samples (existence_0-30)."""
    return dataset.filter(lambda x: x["sub_task"] == "existence_0-30")


def filter_medium(dataset):
    """Keep only medium-object samples (existence_30-60)."""
    return dataset.filter(lambda x: x["sub_task"] == "existence_30-60")


def filter_large(dataset):
    """Keep only large-object samples (existence_60-100)."""
    return dataset.filter(lambda x: x["sub_task"] == "existence_60-100")
