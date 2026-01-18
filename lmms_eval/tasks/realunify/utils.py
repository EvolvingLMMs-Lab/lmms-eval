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


# ============================================================================
# Visual CoT Versions
# ============================================================================

# Mental Reconstruction: 图片被打乱，需要恢复
MENTAL_RECONSTRUCTION_GEN_PROMPT = (
    "Please restore the image that has been shuffled by patches, "
    "without adding extra content or altering the original image."
)

# Attentional Focusing: 高亮与问题相关的区域
ATTENTIONAL_FOCUSING_GEN_PROMPT_TEMPLATE = (
    "Here is the question: {question}\n"
    "Please highlight the regions of the image that are relevant to the question."
)

# Mental Tracking: 根据问题对图片内容进行变换
MENTAL_TRACKING_GEN_PROMPT_TEMPLATE = (
    "Here is the question: {question}\n"
    "Please apply the corresponding transformations and modifications "
    "to the contents of the image according to the question."
)


def doc_to_text_mental_reconstruction_cot(
    doc: Dict, lmms_eval_specific_kwargs: Dict = None
) -> str:
    """Visual CoT prompt for Mental Reconstruction task."""
    prompt = doc.get("evaluation_prompt", doc.get("prompt", "")).strip()

    question_with_aux = (
        "In addition to the original image, you are also given a restored version "
        "of the shuffled image to help you answer the question.\n\n"
        + prompt
    )

    # Add pre_prompt and post_prompt
    if lmms_eval_specific_kwargs:
        if lmms_eval_specific_kwargs.get("pre_prompt"):
            question_with_aux = (
                lmms_eval_specific_kwargs["pre_prompt"] + question_with_aux
            )
        if lmms_eval_specific_kwargs.get("post_prompt"):
            question_with_aux = (
                question_with_aux + lmms_eval_specific_kwargs["post_prompt"]
            )

    return (
        f"[GEN_PROMPT]{MENTAL_RECONSTRUCTION_GEN_PROMPT}[/GEN_PROMPT]"
        f"[QUESTION]{question_with_aux}[/QUESTION]"
    )


def doc_to_text_attentional_focusing_cot(
    doc: Dict, lmms_eval_specific_kwargs: Dict = None
) -> str:
    """Visual CoT prompt for Attentional Focusing task."""
    question = doc.get("question", "").strip()
    prompt = doc.get("evaluation_prompt", doc.get("prompt", "")).strip()

    gen_prompt = ATTENTIONAL_FOCUSING_GEN_PROMPT_TEMPLATE.format(question=question)

    question_with_aux = (
        "In addition to the original image, you are also given a visualization "
        "that highlights the regions relevant to the question.\n\n"
        + prompt
    )

    # Add pre_prompt and post_prompt
    if lmms_eval_specific_kwargs:
        if lmms_eval_specific_kwargs.get("pre_prompt"):
            question_with_aux = (
                lmms_eval_specific_kwargs["pre_prompt"] + question_with_aux
            )
        if lmms_eval_specific_kwargs.get("post_prompt"):
            question_with_aux = (
                question_with_aux + lmms_eval_specific_kwargs["post_prompt"]
            )

    return f"[GEN_PROMPT]{gen_prompt}[/GEN_PROMPT][QUESTION]{question_with_aux}[/QUESTION]"


def doc_to_text_mental_tracking_cot(
    doc: Dict, lmms_eval_specific_kwargs: Dict = None
) -> str:
    """Visual CoT prompt for Mental Tracking task."""
    question = doc.get("question", "").strip()
    prompt = doc.get("evaluation_prompt", doc.get("prompt", "")).strip()

    gen_prompt = MENTAL_TRACKING_GEN_PROMPT_TEMPLATE.format(question=question)

    question_with_aux = (
        "In addition to the original image, you are also given a transformed version "
        "of the image with the modifications applied according to the question.\n\n"
        + prompt
    )

    # Add pre_prompt and post_prompt
    if lmms_eval_specific_kwargs:
        if lmms_eval_specific_kwargs.get("pre_prompt"):
            question_with_aux = (
                lmms_eval_specific_kwargs["pre_prompt"] + question_with_aux
            )
        if lmms_eval_specific_kwargs.get("post_prompt"):
            question_with_aux = (
                question_with_aux + lmms_eval_specific_kwargs["post_prompt"]
            )

    return f"[GEN_PROMPT]{gen_prompt}[/GEN_PROMPT][QUESTION]{question_with_aux}[/QUESTION]"
