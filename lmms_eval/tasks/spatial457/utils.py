"""Utility functions for Spatial457 benchmark evaluation.

This module provides functions for processing the Spatial457 benchmark,
which evaluates 6D spatial reasoning capabilities of large multimodal models.
"""

import re
from collections import defaultdict
from typing import Any, Optional

from loguru import logger as eval_logger
from PIL import Image

TASK_INSTRUCTIONS: dict[str, str] = {
    "L1_single": ("You are an intelligent chatbot designed to answer questions based on " "an image. Your task is to analyze the images, identify attributes of " "the objects, and then determine the answer to the question.\n"),
    "L2_objects": ("You are an intelligent chatbot designed to answer questions based on " "an image. Your task is to analyze the images, identify attributes of " "multiple objects, and then determine the answer to the question.\n"),
    "L3_2d_spatial": (
        "You are an intelligent chatbot designed to answer questions based on "
        "an image. Your task is to analyze the images, identify attributes of "
        "multiple objects and their spatial relationship from 2D projected "
        "camera view, and then determine the answer to the question.\n"
    ),
    "L4_occ": (
        "You are an intelligent chatbot designed to answer questions based on "
        "an image. Your task is to analyze the images, identify attributes of "
        "multiple objects and their occlusion relationships, and then "
        "determine the answer to the question.\n"
    ),
    "L4_pose": (
        "You are an intelligent chatbot designed to answer questions based on "
        "an image. Your task is to analyze the images, identify attributes of "
        "multiple objects and their facing direction in 3D space from the "
        "camera view, and then determine the answer to the question.\n"
    ),
    "L5_6d_spatial": (
        "You are an intelligent chatbot designed to answer questions based on "
        "an image. Your task is to analyze the images, identify attributes of "
        "multiple objects and their spatial relationship from objects' "
        "perspective in 3D space, and then determine the answer to the "
        "question.\n"
    ),
    "L5_collision": (
        "You are an intelligent chatbot designed to answer questions based on "
        "an image. Your task is to analyze the images, identify attributes of "
        "multiple objects and their potential collision given the assumption "
        "of moving direction in 3D space, and then determine the answer to "
        "the question.\n"
    ),
}

REASONING_INSTRUCTION = (
    "First, you should identify the related objects referred in the questions, "
    "including their shape, color, size; then add a brief reasoning process "
    "about the questions. Each object in the image has a shape "
    "(e.g., 'airliner'), a size (only can be 'small' or 'large'), a color "
    "(e.g. 'blue'). The size of the object is either 'small' or 'large'. "
    "The color of the object is one of the following: 'gray', 'blue', "
    "'purple', 'brown', 'green', 'cyan', 'red', 'yellow'. The direction of "
    "the object is one of the following: 'left', 'right', 'front', 'back'.\n\n"
    "Second, give the answer based on the reasoning process. The answer should "
    "only be (1) a phrase, or (2) an integer [0-10] when asked for 'How many' "
    "or 'What is the number of', or (3) 'Yes' or 'No' when asked for "
    "'Is there'. If you think there are no possible answers or the question "
    "is not clear, choose the best answer that fits the question.\n\n"
    "Write your response into this json template: "
    "{'Reasoning': '<your reasons>', 'Answer': '<Your answer>'}"
)


def spatial457_doc_to_visual(doc: dict[str, Any]) -> list[Image.Image]:
    """Extract visual content from a Spatial457 document.

    Args:
        doc: Document containing image information.

    Returns:
        List containing a single PIL Image in RGB format.
    """
    image = doc.get("image")
    if image is None:
        return []
    if isinstance(image, Image.Image):
        return [image.convert("RGB")]
    return [image]


def spatial457_doc_to_text(
    doc: dict[str, Any],
    lmms_eval_specific_kwargs: Optional[dict[str, Any]] = None,
) -> str:
    """Convert a document to prompt text for Spatial457.

    Args:
        doc: Document containing question and metadata.
        lmms_eval_specific_kwargs: Task-specific kwargs including category,
            pre_prompt, and post_prompt.

    Returns:
        Formatted prompt string with task instruction, question, and reasoning
        instruction.
    """
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    question = doc.get("question", "")

    # Use category from task config if available, otherwise infer from doc
    category = lmms_eval_specific_kwargs.get("category", _get_category_from_doc(doc))

    task_instruction = TASK_INSTRUCTIONS.get(category, "")
    prompt = task_instruction + question + "\n" + REASONING_INSTRUCTION

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    return pre_prompt + prompt + post_prompt


def spatial457_process_results(
    doc: dict[str, Any],
    results: list[str],
    lmms_eval_specific_kwargs: Optional[dict[str, Any]] = None,
) -> dict[str, dict[str, Any]]:
    """Process model results for Spatial457 evaluation.

    Args:
        doc: Document containing ground truth answer.
        results: List of model responses.
        lmms_eval_specific_kwargs: Task-specific kwargs including category.

    Returns:
        Dictionary with accuracy metrics and detailed result data.
    """
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    response = results[0] if results else ""
    gt_answer = str(doc.get("answer", "")).strip()

    # Use category from task config if available, otherwise infer from doc
    category = lmms_eval_specific_kwargs.get("category", _get_category_from_doc(doc))
    question_index = doc.get("question_index", 0)

    pred_answer = _extract_answer_from_response(response)
    is_correct = _check_correctness(gt_answer, pred_answer)

    result_data = {
        "question_index": question_index,
        "category": category,
        "gt_answer": gt_answer,
        "pred_answer": pred_answer,
        "response": response,
        "is_correct": is_correct,
    }

    return {"spatial457_accuracy": result_data}


def spatial457_aggregate_results(results: list[dict[str, Any]]) -> float:
    """Aggregate Spatial457 evaluation results.

    Args:
        results: List of individual result dictionaries.

    Returns:
        Overall accuracy as a float between 0 and 1.
    """
    category_scores: dict[str, list[int]] = defaultdict(list)
    total_correct = 0
    total_samples = len(results)

    for result in results:
        category = result["category"]
        is_correct = result["is_correct"]
        score = 1 if is_correct else 0
        category_scores[category].append(score)
        total_correct += score

    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    eval_logger.info("Spatial457 Results")
    eval_logger.info("-" * 50)
    eval_logger.info(f"{'Category':<20} {'Accuracy':>10} {'Count':>10}")
    eval_logger.info("-" * 50)

    for category in sorted(category_scores.keys()):
        scores = category_scores[category]
        cat_accuracy = sum(scores) / len(scores) if scores else 0.0
        eval_logger.info(f"{category:<20} {cat_accuracy:>10.4f} {len(scores):>10}")

    eval_logger.info("-" * 50)
    eval_logger.info(f"{'Overall':<20} {overall_accuracy:>10.4f} {total_samples:>10}")

    return overall_accuracy


def _get_category_from_doc(doc: dict[str, Any]) -> str:
    """Infer category from document.

    Note: This fallback method uses question_index thresholds and may not work
    correctly with HuggingFace datasets where question indices don't follow
    the expected pattern. Prefer passing category via lmms_eval_specific_kwargs.

    Args:
        doc: Document containing category or question_index.

    Returns:
        Category string (e.g., 'L1_single', 'L2_objects').
    """
    if "category" in doc:
        return doc["category"]

    question_index = doc.get("question_index", 0)
    if question_index < 200000:
        return "L1_single"
    elif question_index < 300000:
        return "L2_objects"
    elif question_index < 400000:
        return "L3_2d_spatial"
    elif question_index < 450000:
        return "L4_occ"
    elif question_index < 500000:
        return "L4_pose"
    elif question_index < 550000:
        return "L5_6d_spatial"
    else:
        return "L5_collision"


def _extract_answer_from_response(text: str) -> str:
    """Extract the answer from a model response.

    Args:
        text: Raw model response text.

    Returns:
        Extracted answer string.
    """
    patterns = [
        r"['\"]?Answer['\"]?\s*[:=]\s*['\"]([^'\"]+)['\"]",
        r"Answer['\"]?\s*:\s*['\"]?(\w+)",
        r"['\"]Answer['\"]:\s*(\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return text.strip()


def _check_correctness(gt: str, pred: str) -> bool:
    """Check if prediction matches ground truth.

    Handles various equivalent representations (yes/true/1, no/false/0).

    Args:
        gt: Ground truth answer.
        pred: Predicted answer.

    Returns:
        True if answers match, False otherwise.
    """
    gt_normalized = gt.strip().lower()
    pred_normalized = pred.strip().lower()

    if gt_normalized == pred_normalized:
        return True

    yes_variants = {"yes", "true", "1"}
    no_variants = {"no", "false", "0"}

    if gt_normalized in yes_variants and pred_normalized in yes_variants:
        return True
    if gt_normalized in no_variants and pred_normalized in no_variants:
        return True

    if gt_normalized.isdigit() and pred_normalized.isdigit():
        return int(gt_normalized) == int(pred_normalized)

    return False
