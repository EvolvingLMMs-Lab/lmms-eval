# VideoMME Revert OE MCQ Setting Utils
# Convert MCQ to open-ended question (remove options)

import re

from lmms_eval.tasks.videomme.utils import (
    videomme_doc_to_visual,
)


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Build open-ended question text (without options)."""
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "\nPlease answer the question with a short answer.")

    question = doc["question"]
    full_prompt = f"{pre_prompt}{question}{post_prompt}"
    return full_prompt


def doc_to_target(doc):
    """Get the content of the correct answer as target."""
    answer = doc["answer"]
    options = doc["options"]

    # Get answer index (A=0, B=1, etc.)
    answer_idx = ord(answer.upper()) - ord("A")

    if 0 <= answer_idx < len(options):
        opt = options[answer_idx]
        # Extract content after "X. "
        if ". " in opt:
            return opt.split(". ", 1)[1].strip()
        return opt.strip()

    return answer


def _normalize_text(text):
    """Normalize text for comparison."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def process_results(doc, results):
    """Process results for open-ended format."""
    if not results:
        return {"acc_score": 0.0}

    target = doc_to_target(doc)
    target_normalized = _normalize_text(target)

    acc_score = 0.0
    for pred in results:
        pred_normalized = _normalize_text(pred)
        # Check if target is contained in prediction or vice versa
        if target_normalized in pred_normalized or pred_normalized in target_normalized:
            acc_score += 1.0

    return {"acc_score": acc_score / len(results)}
