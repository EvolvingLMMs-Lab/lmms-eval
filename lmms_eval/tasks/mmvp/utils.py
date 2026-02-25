"""
MMVP (Multimodal Visual Patterns) Task Utils

MMVP is a benchmark that focuses on identifying "CLIP-blind pairs" - images perceived
as similar by CLIP despite clear visual differences. It tests VLMs across 9 basic
visual patterns including orientation, direction, color, counting, etc.

The dataset contains 300 samples (150 pairs) where each pair consists of two images
with the same question but opposite correct answers (A and B).

Ground Truth Corrections (applied in lmms-lab-eval/MMVP dataset):
Based on verification in https://github.com/EvolvingLMMs-Lab/lmms-eval/issues/1018
and the follow-up comment, the following corrections were applied directly to the
dataset (using 1-based Index column values):
- Index 99 (row 98, 0-based): Corrected from (a) to (b) (elephant has short tusks)
- Index 100 (row 99, 0-based): Corrected from (b) to (a) (elephant has long tusks)
- Index 279 (row 278, 0-based): Corrected from (a) to (b) (person is sitting)
- Index 280 (row 279, 0-based): Corrected from (b) to (a) (person is standing)

References:
- Original MMVP: https://github.com/tsb0601/MMVP
- HuggingFace Dataset: https://huggingface.co/datasets/lmms-lab-eval/MMVP
- Original Issue: https://github.com/tsb0601/MMVP/issues/30
"""

import re
from typing import Any, Dict, List, Optional

from loguru import logger as eval_logger


def _extract_answer_letter(text: str) -> str:
    """
    Extract the answer choice letter (A or B) from a string.

    Examples:
        'A' -> 'A'
        'A.' -> 'A'
        '(A)' -> 'A'
        'A answer' -> 'A'
        'The answer is A' -> 'A'
        'Option A' -> 'A'

    Returns:
        The extracted letter (uppercase) or empty string if no letter found.
    """
    text = text.strip()

    # Try common patterns
    patterns = [
        r"^\s*\(?([AB])\)?\.?\s*$",  # Just the letter with optional parens/period
        r"^\s*\(?([AB])\)?[\.\s]",  # Letter at start
        r"answer\s+is\s+\(?([AB])\)?",  # "answer is A/B"
        r"option\s+\(?([AB])\)?",  # "option A/B"
        r"\(?([AB])\)?(?:\s|$)",  # Any standalone A/B
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # Fallback: check first character
    if text and text[0].upper() in ["A", "B"]:
        return text[0].upper()

    return ""


def mmvp_doc_to_visual(doc: Dict[str, Any]) -> List:
    """
    Extract the image from the document.

    Args:
        doc: A sample from the MMVP dataset

    Returns:
        A list containing the image
    """
    image = doc["image"]
    if hasattr(image, "convert"):
        image = image.convert("RGB")
    return [image]


def _parse_options(options_str: str) -> List[str]:
    """Parse options from format '(a) Option1 (b) Option2' to list ['Option1', 'Option2']."""
    pattern = r"\([a-z]\)\s*([^(]+?)(?=\s*\([a-z]\)|$)"
    matches = re.findall(pattern, options_str, re.IGNORECASE)
    return [m.strip() for m in matches]


def mmvp_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> str:
    """
    Construct the prompt text from the document.

    The MMVP dataset has binary choice questions (A/B) about visual patterns.

    Args:
        doc: A sample from the MMVP dataset
        lmms_eval_specific_kwargs: Additional kwargs including pre_prompt and post_prompt

    Returns:
        The formatted prompt string
    """
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    question = doc.get("Question", doc.get("question", ""))
    options_raw = doc.get("Options", doc.get("options", ""))

    if isinstance(options_raw, str):
        options = _parse_options(options_raw)
    else:
        options = options_raw

    options_str = "\n".join([f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)])

    prompt = f"{pre_prompt}{question}\n{options_str}{post_prompt}"
    return prompt


def _normalize_answer(answer: str) -> str:
    """Convert answer format from '(a)' or '(b)' to 'A' or 'B'."""
    answer = answer.strip().lower()
    if "(a)" in answer or answer == "a":
        return "A"
    if "(b)" in answer or answer == "b":
        return "B"
    return answer.upper()


def mmvp_doc_to_target(doc: Dict[str, Any]) -> str:
    """
    Get the target answer for the document.

    Args:
        doc: A sample from the MMVP dataset

    Returns:
        The target answer (A or B)
    """
    original_answer = doc.get("Correct Answer", doc.get("answer", ""))
    return _normalize_answer(original_answer)


def mmvp_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    """
    Process the model prediction and compute correctness.

    Args:
        doc: A sample from the MMVP dataset
        results: List containing the model's prediction

    Returns:
        Dictionary with metrics
    """
    pred = results[0]
    index = doc.get("Index", doc.get("index", doc.get("idx", -1)))
    if isinstance(index, int) and index > 0:
        index = index - 1

    gt = mmvp_doc_to_target(doc)
    pred_letter = _extract_answer_letter(pred)
    is_correct = pred_letter == gt

    pair_index = index // 2

    result = {
        "index": index,
        "pair_index": pair_index,
        "pred": pred,
        "pred_letter": pred_letter,
        "gt": gt,
        "is_correct": is_correct,
    }

    return {
        "mmvp_accuracy": result,
        "mmvp_pair_accuracy": result,
    }


def mmvp_aggregate_results(results: List[Dict[str, Any]]) -> float:
    """
    Aggregate results to compute overall accuracy.

    Args:
        results: List of result dictionaries from process_results

    Returns:
        The accuracy as a float between 0 and 1
    """
    if not results:
        return 0.0

    correct = sum(1 for r in results if r["is_correct"])
    total = len(results)
    accuracy = correct / total

    eval_logger.info(f"MMVP Accuracy: {accuracy:.4f} ({correct}/{total})")

    return accuracy


def mmvp_aggregate_pair_results(results: List[Dict[str, Any]]) -> float:
    """
    Aggregate results to compute pair accuracy.

    In MMVP, each pair of consecutive samples (0-1, 2-3, etc.) tests the same
    visual pattern with opposite correct answers. Pair accuracy measures how
    often the model gets BOTH samples in a pair correct.

    This is a stricter metric that better captures genuine visual understanding
    vs. lucky guessing.

    Args:
        results: List of result dictionaries from process_results

    Returns:
        The pair accuracy as a float between 0 and 1
    """
    if not results:
        return 0.0

    # Group results by pair index
    pairs: Dict[int, List[Dict[str, Any]]] = {}
    for r in results:
        pair_idx = r["pair_index"]
        if pair_idx not in pairs:
            pairs[pair_idx] = []
        pairs[pair_idx].append(r)

    # Count pairs where both samples are correct
    correct_pairs = 0
    total_pairs = 0

    for pair_idx, pair_results in pairs.items():
        if len(pair_results) == 2:
            total_pairs += 1
            if all(r["is_correct"] for r in pair_results):
                correct_pairs += 1
        elif len(pair_results) == 1:
            # Handle edge case of incomplete pairs (shouldn't happen with full dataset)
            eval_logger.warning(f"MMVP: Incomplete pair at index {pair_idx}, " f"only {len(pair_results)} sample(s)")

    if total_pairs == 0:
        return 0.0

    pair_accuracy = correct_pairs / total_pairs
    eval_logger.info(f"MMVP Pair Accuracy: {pair_accuracy:.4f} ({correct_pairs}/{total_pairs})")

    return pair_accuracy
