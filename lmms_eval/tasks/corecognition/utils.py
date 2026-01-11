"""Utility functions for CoreCognition benchmark."""

import re
from collections import defaultdict
from typing import Any, Dict, Optional


def corecognition_doc_to_visual(doc: dict[str, Any]) -> list:
    """Extract image from document.
    Args:
        doc: Document containing images field
    Returns:
        List containing the RGB image
    """
    return [doc["images"].convert("RGB")]


def corecognition_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, str]] = None) -> str:
    """Format question text with optional prompt additions.
    Args:
        doc: Document containing prompt field
        lmms_eval_specific_kwargs: Optional pre/post prompts
    Returns:
        Formatted question string
    """
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    prompt = doc.get("prompt", "")

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    return f"{pre_prompt}{prompt}{post_prompt}"


def filter_stage_sensorimotor(doc: dict[str, Any]) -> bool:
    """Filter for Stage Sensorimotor samples only."""
    return doc.get("stage", "") == "Stage Sensorimotor"


def filter_stage_concrete_operational(doc: dict[str, Any]) -> bool:
    """Filter for Stage Concrete Operational samples only."""
    return doc.get("stage", "") == "Stage Concrete Operational"


def filter_stage_formal_operational(doc: dict[str, Any]) -> bool:
    """Filter for Stage Formal Operational samples only."""
    return doc.get("stage", "") == "Stage Formal Operational"


def corecognition_process_results(doc: dict[str, Any], results: list[str]) -> dict[str, Any]:
    """Process model results and compute accuracy.
    Args:
        doc: Document containing ground truth answer
        results: List containing model prediction
    Returns:
        Dictionary with accuracy metric
    """
    pred = results[0] if isinstance(results[0], str) else str(results[0])
    ground_truth = str(doc["answer"]).strip().upper()
    concept = doc.get("concept", "unknown")

    pred = pred.strip()
    if not pred:
        pred_normalized = ""
    else:
        pred_normalized = _extract_answer(pred)

    is_correct = pred_normalized == ground_truth

    return {
        "accuracy": float(is_correct),
        "accuracy_by_concept": {"concept": concept, "correct": is_correct},
    }


def _extract_answer(pred: str) -> str:
    """Extract answer from model prediction with aggressive cleanup.
    Args:
        pred: Raw model prediction
    Returns:
        Extracted answer in uppercase
    """
    pred = pred.strip()
    
    patterns = [
        r'^(yes|no|[a-d])(\.|\,|\;| |\n|\*)',
        r'[\n\*]+(yes|no|[a-d])(\.|\,|\;| |\n|\*)',
        r'(yes|no|[a-d]) is the correct answer',
        r'answer is[\:\;\*\n ]*(yes|no|[a-d])',
        r'answer[\:\;\*\n ]*(yes|no|[a-d])',
        r'option is[\:\;\*\n ]*(yes|no|[a-d])',
        r'choice is[\:\;\*\n ]*(yes|no|[a-d])',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, pred, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    cleaned = re.split(r'[,\.\:\;\n\s]+', pred)[0].strip()
    if cleaned:
        return cleaned.upper()
    
    return pred.upper()


def corecognition_aggregate_by_concept(
    results: list[dict[str, Any]],
) -> dict[str, float]:
    """Aggregate results by concept.
    Args:
        results: List of result dictionaries with concept and correctness
    Returns:
        Dictionary mapping concept names to accuracy scores
    """
    concept_correct: dict[str, int] = defaultdict(int)
    concept_total: dict[str, int] = defaultdict(int)

    for result in results:
        concept = result["concept"]
        correct = result["correct"]

        concept_total[concept] += 1
        if correct:
            concept_correct[concept] += 1

    concept_accuracy = {}
    for concept in concept_total:
        accuracy = concept_correct[concept] / concept_total[concept]
        concept_accuracy[concept] = accuracy

    total_correct = sum(concept_correct.values())
    total = sum(concept_total.values())
    concept_accuracy["overall"] = total_correct / total if total > 0 else 0.0

    return concept_accuracy
