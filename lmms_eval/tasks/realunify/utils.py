"""RealUnify Task Utilities.

GEU (Generation Enhances Understanding) evaluation tasks.
Paper: https://arxiv.org/abs/2509.24897
"""

import re
from collections import defaultdict
from io import BytesIO
from typing import Any

from loguru import logger as eval_logger
from PIL import Image


def doc_to_visual(doc: dict) -> list[Image.Image]:
    """Extract visual input from document."""
    if "image" in doc and doc["image"]:
        img_data = doc["image"]
        if isinstance(img_data, Image.Image):
            return [img_data.convert("RGB")]
        elif isinstance(img_data, bytes):
            return [Image.open(BytesIO(img_data)).convert("RGB")]
        elif isinstance(img_data, dict) and "bytes" in img_data:
            return [Image.open(BytesIO(img_data["bytes"])).convert("RGB")]
    return []


def doc_to_text(doc: dict, lmms_eval_specific_kwargs: dict | None = None) -> str:
    """Format evaluation prompt from document."""
    prompt = doc.get("evaluation_prompt", doc.get("prompt", "")).strip()

    if lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
        if pre_prompt:
            prompt = pre_prompt + prompt
        if post_prompt:
            prompt = prompt + post_prompt

    return prompt


def extract_answer(response: str) -> str:
    """Extract answer letter (A/B/C/D) from model response."""
    response = response.strip()

    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
        "Answer:",
        "Answer is",
        "Answer",
    ]
    for prefix in answer_prefixes:
        if response.lower().startswith(prefix.lower()):
            response = response[len(prefix) :].strip()

    if len(response.split()) > 10 and not re.search(r"[ABCD]", response):
        return ""

    match = re.search(r"[ABCD]", response.upper())
    if match:
        return match.group(0)
    return ""


def process_results(doc: dict, results: list[str]) -> dict[str, Any]:
    """Process model results and compute accuracy."""
    pred_text = results[0] if results else ""
    pred_answer = extract_answer(pred_text)
    gt_answer = doc.get("answer", "").strip().upper()
    score = 1.0 if pred_answer == gt_answer else 0.0
    task_type = doc.get("task_type", doc.get("category", "unknown"))

    return {
        "mental_tracking": {"task_type": task_type, "score": score},
        "mental_reconstruction": {"task_type": task_type, "score": score},
        "attentional_focusing": {"task_type": task_type, "score": score},
        "accuracy": {"task_type": task_type, "score": score},
    }


def aggregate_results(results: list[dict]) -> float:
    """Aggregate results to compute overall accuracy."""
    task_scores: dict[str, list[float]] = defaultdict(list)

    for result in results:
        score = result["score"]
        task_type = result.get("task_type", "unknown")
        task_scores[task_type].append(score)

    for task_type, scores in sorted(task_scores.items()):
        avg = sum(scores) / len(scores) if scores else 0.0
        eval_logger.info(f"  {task_type}: {avg:.4f} ({sum(scores):.0f}/{len(scores)})")

    all_scores = [s for scores in task_scores.values() for s in scores]
    return sum(all_scores) / len(all_scores) if all_scores else 0.0


MENTAL_RECONSTRUCTION_GEN_PROMPT = "Please restore the image that has been shuffled by patches, " "without adding extra content or altering the original image."

ATTENTIONAL_FOCUSING_GEN_PROMPT_TEMPLATE = "Here is the question: {question}\n" "Please highlight the regions of the image that are relevant to the question."

MENTAL_TRACKING_GEN_PROMPT_TEMPLATE = "Here is the question: {question}\n" "Please apply the corresponding transformations and modifications " "to the contents of the image according to the question."


def doc_to_text_mental_reconstruction_cot(doc: dict, lmms_eval_specific_kwargs: dict | None = None) -> str:
    """Visual CoT prompt for Mental Reconstruction task."""
    prompt = doc.get("evaluation_prompt", doc.get("prompt", "")).strip()

    question_with_aux = "In addition to the original image, you are also given a restored version " "of the shuffled image to help you answer the question.\n\n" + prompt

    if lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
        if pre_prompt:
            question_with_aux = pre_prompt + question_with_aux
        if post_prompt:
            question_with_aux = question_with_aux + post_prompt

    return f"[GEN_PROMPT]{MENTAL_RECONSTRUCTION_GEN_PROMPT}[/GEN_PROMPT]" f"[QUESTION]{question_with_aux}[/QUESTION]"


def doc_to_text_attentional_focusing_cot(doc: dict, lmms_eval_specific_kwargs: dict | None = None) -> str:
    """Visual CoT prompt for Attentional Focusing task."""
    question = doc.get("question", "").strip()
    prompt = doc.get("evaluation_prompt", doc.get("prompt", "")).strip()

    gen_prompt = ATTENTIONAL_FOCUSING_GEN_PROMPT_TEMPLATE.format(question=question)

    question_with_aux = "In addition to the original image, you are also given a visualization " "that highlights the regions relevant to the question.\n\n" + prompt

    if lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
        if pre_prompt:
            question_with_aux = pre_prompt + question_with_aux
        if post_prompt:
            question_with_aux = question_with_aux + post_prompt

    return f"[GEN_PROMPT]{gen_prompt}[/GEN_PROMPT][QUESTION]{question_with_aux}[/QUESTION]"


def doc_to_text_mental_tracking_cot(doc: dict, lmms_eval_specific_kwargs: dict | None = None) -> str:
    """Visual CoT prompt for Mental Tracking task."""
    question = doc.get("question", "").strip()
    prompt = doc.get("evaluation_prompt", doc.get("prompt", "")).strip()

    gen_prompt = MENTAL_TRACKING_GEN_PROMPT_TEMPLATE.format(question=question)

    question_with_aux = "In addition to the original image, you are also given a transformed version " "of the image with the modifications applied according to the question.\n\n" + prompt

    if lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
        if pre_prompt:
            question_with_aux = pre_prompt + question_with_aux
        if post_prompt:
            question_with_aux = question_with_aux + post_prompt

    return f"[GEN_PROMPT]{gen_prompt}[/GEN_PROMPT][QUESTION]{question_with_aux}[/QUESTION]"
