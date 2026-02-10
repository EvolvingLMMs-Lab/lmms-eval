"""
3DSRBench: A Benchmark for 3D Spatial Reasoning in Vision-Language Models.

This module provides utilities for evaluating vision-language models on the 3DSRBench dataset.
The evaluation includes multiple accuracy metrics:
- vanilla_accuracy: Raw accuracy on all questions
- flip_accuracy: Accuracy where both original and answer-flipped questions must be correct
- circular_accuracy: Accuracy where all circular variants must be correct
- flip_circular_accuracy: Most strict - all variants including flip must be correct
- category-specific accuracy: height, location, orientation, multi_object
"""

import re
import string
from typing import Optional

import pandas as pd
from loguru import logger as eval_logger

# Category mapping from detailed categories to main categories
CATEGORY_MAPPING = {
    "height_higher": "height",
    "location_above": "location",
    "location_closer_to_camera": "location",
    "location_next_to": "location",
    "orientation_in_front_of": "orientation",
    "orientation_on_the_left": "orientation",
    "orientation_viewpoint": "orientation",
    "multi_object_closer_to": "multi_object",
    "multi_object_facing": "multi_object",
    "multi_object_viewpoint_towards_object": "multi_object",
    "multi_object_parallel": "multi_object",
    "multi_object_same_direction": "multi_object",
}

MAIN_CATEGORIES = ["height", "location", "orientation", "multi_object"]


def doc_to_visual(doc):
    """Extract the image from the document.

    Args:
        doc: A document from the 3DSRBench dataset.

    Returns:
        list: A list containing the RGB-converted image.
    """
    return [doc["image"].convert("RGB")]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Build the prompt text from the document.

    Following VLMEvalKit's format:
    - Hint (if present)
    - Question
    - Options (A, B, C, D)
    - Post prompt asking to select correct answer

    Args:
        doc: A document from the 3DSRBench dataset.
        lmms_eval_specific_kwargs: Optional kwargs including pre_prompt and post_prompt.

    Returns:
        str: The formatted prompt text.
    """
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "Please select the correct answer from the options above. \n")

    question = doc["question"]

    # Build options from A, B, C, D columns
    options = {}
    for cand in string.ascii_uppercase[:4]:  # A, B, C, D
        if cand in doc and doc[cand] is not None:
            val = doc[cand]
            # Skip 'nan' string values (some options might be empty)
            if isinstance(val, str) and val.lower() == "nan":
                continue
            if pd.isna(val):
                continue
            options[cand] = val

    options_prompt = "Options:\n"
    for key, item in options.items():
        options_prompt += f"{key}. {item}\n"

    # Build the full prompt following VLMEvalKit format
    prompt = ""
    if pre_prompt:
        prompt += pre_prompt
    prompt += f"Question: {question}\n"
    if options:
        prompt += options_prompt
    if post_prompt:
        prompt += post_prompt

    return prompt


def extract_answer(text: str) -> Optional[str]:
    """Extract the answer letter (A, B, C, D) from the model's response.

    Uses multiple regex patterns to find the answer in various formats.

    Args:
        text: The model's response text.

    Returns:
        The extracted answer letter (A-D), or None if not found.
    """
    if not text:
        return None

    text = text.strip()

    # Try various patterns to extract the answer
    patterns = [
        # Direct answer patterns
        r"^([A-D])[\.\s\)]",  # Starts with A. or A) or A
        r"^([A-D])$",  # Just the letter
        r"[Aa]nswer[:\s]+([A-D])",  # "Answer: A" or "answer A"
        r"[Tt]he answer is[:\s]+([A-D])",  # "The answer is A"
        r"[Mm]y answer is[:\s]+([A-D])",  # "My answer is A"
        r"\(([A-D])\)",  # (A)
        r"([A-D])\.",  # A.
        r"\b([A-D])\b",  # Any standalone A, B, C, D
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()

    return None


def get_qid_key(qid: str, mode: str = "base") -> str:
    """Extract the base question ID for grouping.

    The qid format in 3dsr_circular is:
    - "XXXXXXXX" (base question, e.g., "VIN6MS3J")
    - "XXXXXXXX-N" (circular variant, e.g., "VIN6MS3J-1")
    - "XXXXXXXX-flip" (flipped question, e.g., "VIN6MS3J-flip")
    - "XXXXXXXX-flip-N" (flipped circular variant, e.g., "VIN6MS3J-flip-1")

    Args:
        qid: The question ID.
        mode: The grouping mode:
            - "base": Return the 8-char base ID (for flip_circular)
            - "flip": Return base or base-flip (for flip eval)
            - "circular": Return base or base-flip (for circular eval without flip grouping)

    Returns:
        The grouped question ID key.
    """
    if mode == "base":
        # Return the first 8 characters (base ID)
        return qid[:8]
    elif mode == "flip":
        # For flip eval: group by base ID, ignoring -N suffix but keeping -flip distinction
        # "VIN6MS3J" and "VIN6MS3J-flip" -> "VIN6MS3J"
        # "VIN6MS3J-1" and "VIN6MS3J-flip-1" -> "VIN6MS3J"
        return qid[:8]
    elif mode == "circular":
        # For circular eval: group circular variants together, but keep flip separate
        # "VIN6MS3J" and "VIN6MS3J-1" -> "VIN6MS3J"
        # "VIN6MS3J-flip" and "VIN6MS3J-flip-1" -> "VIN6MS3J-flip"
        if "-flip" in qid:
            # Return up to 13 chars: base + "-flip"
            return qid[:13]
        else:
            return qid[:8]
    else:
        return qid


def get_main_category(category: str) -> str:
    """Map detailed category to main category.

    Args:
        category: The detailed category (e.g., "height_higher").

    Returns:
        The main category (e.g., "height").
    """
    return CATEGORY_MAPPING.get(category, "other")


def process_results(doc, results):
    """Process the model's output for a single document.

    Args:
        doc: The document from the dataset.
        results: List containing the model's response.

    Returns:
        dict: Results for each metric type.
    """
    pred = results[0].strip()
    pred_answer = extract_answer(pred)
    gt_answer = doc["answer"]

    score = 1.0 if pred_answer == gt_answer else 0.0

    # Get category information
    category = doc.get("category", "unknown")
    main_category = get_main_category(category)

    # Use index as the unique identifier (works for both subsets)
    # For 3dsr_circular, we also have qid for grouping
    index = doc["index"]
    qid = doc.get("qid", index)  # Fall back to index if qid not present

    result_dict = {
        "vanilla_accuracy": {
            "index": index,
            "qid": qid,
            "score": score,
            "category": category,
            "main_category": main_category,
        },
        "flip_accuracy": {
            "index": index,
            "qid": qid,
            "score": score,
            "category": category,
            "main_category": main_category,
        },
        "circular_accuracy": {
            "index": index,
            "qid": qid,
            "score": score,
            "category": category,
            "main_category": main_category,
        },
        "flip_circular_accuracy": {
            "index": index,
            "qid": qid,
            "score": score,
            "category": category,
            "main_category": main_category,
        },
        f"height_accuracy": {
            "index": index,
            "qid": qid,
            "score": score,
            "category": category,
            "main_category": main_category,
        },
        f"location_accuracy": {
            "index": index,
            "qid": qid,
            "score": score,
            "category": category,
            "main_category": main_category,
        },
        f"orientation_accuracy": {
            "index": index,
            "qid": qid,
            "score": score,
            "category": category,
            "main_category": main_category,
        },
        f"multi_object_accuracy": {
            "index": index,
            "qid": qid,
            "score": score,
            "category": category,
            "main_category": main_category,
        },
    }

    return result_dict


def aggregate_vanilla_accuracy(results):
    """Aggregate vanilla accuracy (simple mean of all scores).

    Args:
        results: List of result dicts from process_results.

    Returns:
        float: The average accuracy.
    """
    if not results:
        return 0.0
    total_score = sum(r["score"] for r in results)
    return total_score / len(results)


def aggregate_flip_accuracy(results):
    """Aggregate flip accuracy.

    For flip eval, both the original question and its flipped version must be correct.
    Each question is paired with its flip only (circular variants are separate pairs).

    Example grouping:
    - "VIN6MS3J" + "VIN6MS3J-flip" → one group (must both be correct)
    - "VIN6MS3J-1" + "VIN6MS3J-flip-1" → another group (must both be correct)

    Args:
        results: List of result dicts from process_results.

    Returns:
        float: The flip accuracy.
    """
    if not results:
        return 0.0

    # Group by qid with "-flip" removed (pairs original with its flip)
    qid_groups = {}
    for r in results:
        qid = r["qid"]
        # Remove "-flip" to pair original with flipped version
        # "VIN6MS3J" and "VIN6MS3J-flip" → "VIN6MS3J"
        # "VIN6MS3J-1" and "VIN6MS3J-flip-1" → "VIN6MS3J-1"
        key = qid.replace("-flip", "")

        if key not in qid_groups:
            qid_groups[key] = []
        qid_groups[key].append(r["score"])

    # For each group, all must be correct (both original and flip)
    correct = 0
    total = 0
    for key, scores in qid_groups.items():
        group_correct = 1.0
        for s in scores:
            group_correct *= s
        if group_correct == 1.0:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def aggregate_circular_accuracy(results):
    """Aggregate circular accuracy.

    For circular eval, all circular variants of the same question must be correct.
    The flip variants are evaluated separately.

    Args:
        results: List of result dicts from process_results.

    Returns:
        float: The circular accuracy.
    """
    if not results:
        return 0.0

    # Group by circular key (base or base-flip)
    qid_groups = {}
    for r in results:
        qid = r["qid"]
        # Get circular key: "VIN6MS3J" or "VIN6MS3J-flip"
        key = get_qid_key(qid, mode="circular")

        if key not in qid_groups:
            qid_groups[key] = []
        qid_groups[key].append(r["score"])

    # For each group, all must be correct
    correct = 0
    total = 0
    for key, scores in qid_groups.items():
        group_correct = 1.0
        for s in scores:
            group_correct *= s
        if group_correct == 1.0:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def aggregate_flip_circular_accuracy(results):
    """Aggregate flip+circular accuracy (most strict).

    All variants (original, flipped, all circular) must be correct.

    Args:
        results: List of result dicts from process_results.

    Returns:
        float: The flip+circular accuracy.
    """
    if not results:
        return 0.0

    # Group by base 8-char ID (all variants together)
    qid_groups = {}
    for r in results:
        qid = r["qid"]
        base_id = qid[:8]

        if base_id not in qid_groups:
            qid_groups[base_id] = []
        qid_groups[base_id].append(r["score"])

    # For each group, all must be correct
    correct = 0
    total = 0
    for base_id, scores in qid_groups.items():
        group_correct = 1.0
        for s in scores:
            group_correct *= s
        if group_correct == 1.0:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def aggregate_height_accuracy(results):
    """Aggregate accuracy for height category.

    Args:
        results: List of result dicts from process_results.

    Returns:
        float: The height category accuracy.
    """
    if not results:
        return 0.0

    scores = [r["score"] for r in results if r.get("main_category") == "height"]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def aggregate_location_accuracy(results):
    """Aggregate accuracy for location category.

    Args:
        results: List of result dicts from process_results.

    Returns:
        float: The location category accuracy.
    """
    if not results:
        return 0.0

    scores = [r["score"] for r in results if r.get("main_category") == "location"]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def aggregate_orientation_accuracy(results):
    """Aggregate accuracy for orientation category.

    Args:
        results: List of result dicts from process_results.

    Returns:
        float: The orientation category accuracy.
    """
    if not results:
        return 0.0

    scores = [r["score"] for r in results if r.get("main_category") == "orientation"]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def aggregate_multi_object_accuracy(results):
    """Aggregate accuracy for multi_object category.

    Args:
        results: List of result dicts from process_results.

    Returns:
        float: The multi_object category accuracy.
    """
    if not results:
        return 0.0

    scores = [r["score"] for r in results if r.get("main_category") == "multi_object"]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def aggregate_category_accuracy(results):
    """Aggregate accuracy for a specific category (deprecated, kept for compatibility).

    Args:
        results: List of result dicts from process_results.

    Returns:
        float: The overall accuracy across all categories.
    """
    if not results:
        return 0.0

    # Log category-wise results
    for main_cat in MAIN_CATEGORIES:
        scores = [r["score"] for r in results if r.get("main_category") == main_cat]
        if scores:
            acc = sum(scores) / len(scores)
            eval_logger.info(f"3DSRBench {main_cat}: {acc * 100:.2f}%")

    # Return overall accuracy
    all_scores = [r["score"] for r in results]
    return sum(all_scores) / len(all_scores) if all_scores else 0.0
