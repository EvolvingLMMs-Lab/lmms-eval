"""
RealUnify Task Utilities
Evaluation for GEU (Generation Enhances Understanding) tasks:
- Mental Tracking
- Mental Reconstruction
- Attentional Focusing
"""

import re
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List

from loguru import logger as eval_logger
from PIL import Image


def doc_to_visual(doc: Dict) -> List[Image.Image]:
    """Get visual input from document."""
    if "image" in doc and doc["image"]:
        img_data = doc["image"]
        # Handle different image formats
        if isinstance(img_data, Image.Image):
            return [img_data.convert("RGB")]
        elif isinstance(img_data, bytes):
            return [Image.open(BytesIO(img_data)).convert("RGB")]
        elif isinstance(img_data, dict) and "bytes" in img_data:
            return [Image.open(BytesIO(img_data["bytes"])).convert("RGB")]
    return []


def doc_to_text(doc: Dict, lmms_eval_specific_kwargs: Dict = None) -> str:
    """Format evaluation prompt."""
    # GEU tasks use evaluation_prompt field
    prompt = doc.get("evaluation_prompt", doc.get("prompt", "")).strip()

    # Add pre_prompt and post_prompt
    if lmms_eval_specific_kwargs:
        if lmms_eval_specific_kwargs.get("pre_prompt"):
            prompt = lmms_eval_specific_kwargs["pre_prompt"] + prompt
        if lmms_eval_specific_kwargs.get("post_prompt"):
            prompt = prompt + lmms_eval_specific_kwargs["post_prompt"]

    return prompt


def extract_answer(response: str) -> str:
    """Extract answer letter from model response."""
    response = response.strip()

    # Remove common answer prefixes
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
        "Answer",
        "Answer is",
    ]
    for prefix in answer_prefixes:
        response = response.replace(prefix, "")

    # If response is too long and no clear answer, return empty
    if len(response.split()) > 10 and not re.search("[ABCD]", response):
        return ""

    # Find first A/B/C/D
    match = re.search(r"[ABCD]", response)
    if match:
        return match.group(0)
    return ""


def process_results(doc: Dict, results: List[str]) -> Dict[str, Any]:
    """Process results - extract answer and compare to ground truth."""
    pred_text = results[0] if results else ""

    # Extract predicted answer
    pred_answer = extract_answer(pred_text)

    # Get ground truth
    gt_answer = doc.get("answer", "")

    # Compare
    score = 1.0 if pred_answer == gt_answer else 0.0

    task_type = doc.get("task_type", "unknown")

    return {
        task_type: {
            "score": score,
            "gt": gt_answer,
            "pred": pred_answer,
        },
        "accuracy": {
            "task_type": task_type,
            "score": score,
        },
    }


def aggregate_results(results: List[Dict]) -> float:
    """Aggregate results to compute accuracy."""
    task_scores = defaultdict(list)

    for result in results:
        score = result["score"]
        task_type = result.get("task_type", "unknown")
        task_scores[task_type].append(score)

    # Log per-task accuracies
    for task_type, scores in sorted(task_scores.items()):
        avg = sum(scores) / len(scores) if scores else 0.0
        eval_logger.info(f"  {task_type}: {avg:.4f} ({sum(scores):.0f}/{len(scores)})")

    # Overall accuracy
    all_scores = [s for scores in task_scores.values() for s in scores]
    return sum(all_scores) / len(all_scores) if all_scores else 0.0
