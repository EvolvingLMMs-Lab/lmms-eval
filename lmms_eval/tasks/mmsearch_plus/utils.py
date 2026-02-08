"""Utility functions for MMSearch-Plus VQA task."""

import re
from typing import Any, Dict, List

from loguru import logger as eval_logger

from lmms_eval.tasks.mmsearch_plus.decrypt_utils import decrypt_sample

# Canary string for decryption (full dataset name)
CANARY = "MMSearch-Plus"


def mmsearch_plus_doc_to_visual(doc: Dict[str, Any]) -> List[Any]:
    """
    Extract images from document.

    Args:
        doc: Document containing images

    Returns:
        List of PIL Images
    """
    # Decrypt the document
    doc = decrypt_sample(doc, CANARY)

    images = []
    num_images = doc.get("num_images", 0)

    # Extract images based on num_images field
    for i in range(1, num_images + 1):
        img_key = f"img_{i}"
        if img_key in doc and doc[img_key] is not None:
            try:
                images.append(doc[img_key].convert("RGB"))
            except Exception as e:
                eval_logger.warning(f"Failed to load image {img_key}: {e}")

    if not images:
        eval_logger.warning(f"No images found in document")

    return images


def mmsearch_plus_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Dict[str, Any] = None) -> str:
    """
    Convert document to text prompt.

    Args:
        doc: Document containing question
        lmms_eval_specific_kwargs: Model-specific kwargs

    Returns:
        Formatted question prompt
    """
    # Decrypt the document
    doc = decrypt_sample(doc, CANARY)

    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get(
        "post_prompt",
        "\nAnswer the question using a single word or short phrase.",
    )

    question = doc.get("question", "")
    return f"{pre_prompt}{question}{post_prompt}"


def normalize_answer(answer: str) -> str:
    """
    Normalize answer for comparison.

    Args:
        answer: Raw answer string

    Returns:
        Normalized answer
    """
    # Convert to lowercase
    answer = answer.lower().strip()

    # Remove punctuation
    answer = re.sub(r"[^\w\s]", "", answer)

    # Remove extra whitespace
    answer = " ".join(answer.split())

    return answer


def compute_f1_score(prediction: str, ground_truth: str) -> float:
    """
    Compute token-level F1 score between prediction and ground truth.

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer

    Returns:
        F1 score (0-1)
    """
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return 0.0

    # Calculate precision and recall
    common = set(pred_tokens) & set(gt_tokens)

    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)

    # Calculate F1
    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """
    Compute exact match score (1 or 0).

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer

    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)

    return 1.0 if pred_norm == gt_norm else 0.0


def mmsearch_plus_process_results(doc: Dict[str, Any], result: List[str]) -> Dict[str, Any]:
    """
    Process model results and compute metrics.

    Args:
        doc: Document containing ground truth
        result: Model predictions

    Returns:
        Dictionary containing metrics
    """
    # Decrypt the document
    doc = decrypt_sample(doc, CANARY)

    if not result or len(result) == 0:
        eval_logger.warning("Empty result received")
        return {
            "f1_score": 0.0,
            "exact_match": 0.0,
        }

    prediction = result[0].strip()

    # Get ground truth answers
    gt_answers = doc.get("answer", [])
    if isinstance(gt_answers, str):
        gt_answers = [gt_answers]

    if not gt_answers:
        eval_logger.warning("No ground truth answers found")
        return {
            "f1_score": 0.0,
            "exact_match": 0.0,
        }

    # Compute max score across all valid answers
    max_f1 = 0.0
    max_em = 0.0

    for gt_answer in gt_answers:
        f1 = compute_f1_score(prediction, gt_answer)
        em = compute_exact_match(prediction, gt_answer)

        max_f1 = max(max_f1, f1)
        max_em = max(max_em, em)

    return {
        "f1_score": max_f1,
        "exact_match": max_em,
    }
