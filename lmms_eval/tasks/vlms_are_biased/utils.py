"""Utility functions for VLMs Are Biased benchmark."""

from collections import defaultdict
from typing import Any, Dict, Optional


def vlms_are_biased_doc_to_visual(doc: dict[str, Any]) -> list:
    """Extract image from document.

    Args:
        doc: Document containing image field

    Returns:
        List containing the RGB image
    """
    return [doc["image"].convert("RGB")]


def vlms_are_biased_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, str]] = None) -> str:
    """Format question text with optional prompt additions.

    Args:
        doc: Document containing prompt field
        lmms_eval_specific_kwargs: Optional pre/post prompts

    Returns:
        Formatted question string
    """
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    # Use the prompt field directly from the dataset
    prompt = doc.get("prompt", "")

    # Allow optional pre/post prompts (for experimental variations)
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    return f"{pre_prompt}{prompt}{post_prompt}"


def vlms_are_biased_process_results(doc: dict[str, Any], results: list[str]) -> dict[str, Any]:
    """Process model results and compute accuracy and bias ratio.

    Args:
        doc: Document containing ground truth and expected bias
        results: List containing model prediction

    Returns:
        Dictionary with accuracy and bias_ratio metrics
    """
    pred = results[0].strip()
    ground_truth = str(doc["ground_truth"]).strip()
    expected_bias = str(doc["expected_bias"]).strip()
    topic = doc.get("topic", "unknown")

    # Normalize for comparison (case-insensitive, handle Yes/No in braces)
    pred_normalized = pred.lower().strip("{}").strip()
    gt_normalized = ground_truth.lower().strip("{}").strip()
    bias_normalized = expected_bias.lower().strip("{}").strip()

    # Check for exact match with ground truth
    is_correct = pred_normalized == gt_normalized

    # Also check if prediction matches expected bias (measures bias)
    matches_bias = pred_normalized == bias_normalized

    # Try to extract and compare numbers if exact match fails
    if not is_correct and not matches_bias:
        pred_numbers = "".join(c for c in pred_normalized if c.isdigit())
        gt_numbers = "".join(c for c in gt_normalized if c.isdigit())
        bias_numbers = "".join(c for c in bias_normalized if c.isdigit())

        if pred_numbers and gt_numbers:
            is_correct = pred_numbers == gt_numbers
        if pred_numbers and bias_numbers:
            matches_bias = pred_numbers == bias_numbers

    return {
        "accuracy": float(is_correct),
        "bias_ratio": float(matches_bias),
        "accuracy_by_topic": {"topic": topic, "correct": is_correct},
    }


def vlms_are_biased_aggregate_by_topic(
    results: list[dict[str, Any]],
) -> dict[str, float]:
    """Aggregate results by topic.

    Args:
        results: List of result dictionaries with topic and correctness

    Returns:
        Dictionary mapping topic names to accuracy scores
    """
    topic_correct: dict[str, int] = defaultdict(int)
    topic_total: dict[str, int] = defaultdict(int)

    for result in results:
        topic = result["topic"]
        correct = result["correct"]

        topic_total[topic] += 1
        if correct:
            topic_correct[topic] += 1

    # Calculate accuracy per topic
    topic_accuracy = {}
    for topic in topic_total:
        accuracy = topic_correct[topic] / topic_total[topic]
        topic_accuracy[topic] = accuracy

    # Add overall accuracy
    total_correct = sum(topic_correct.values())
    total = sum(topic_total.values())
    topic_accuracy["overall"] = total_correct / total if total > 0 else 0.0

    return topic_accuracy
