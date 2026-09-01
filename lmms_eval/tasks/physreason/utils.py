"""Utility functions for PhysReason benchmark evaluation.

Handles prompt construction, native Hugging Face image columns, and scoring
for open-ended physics problem solving.

Dataset: https://huggingface.co/datasets/zhibei1204/PhysReason
Paper:   https://arxiv.org/abs/2502.12054
"""

import re
from io import BytesIO

import numpy as np
from loguru import logger as eval_logger
from PIL import Image


def physreason_doc_to_visual(doc):
    """Return every image associated with the problem."""
    visuals = []
    image_columns = [doc.get(f"image_{index}") for index in range(1, 6)]
    images = [image for image in image_columns if image is not None]
    if not images:
        # Retain compatibility with the original nested List(Image) schema.
        images = doc.get("images", [])

    for image in images:
        try:
            if isinstance(image, Image.Image):
                visuals.append(image.convert("RGB"))
            elif isinstance(image, dict) and image.get("bytes") is not None:
                visuals.append(Image.open(BytesIO(image["bytes"])).convert("RGB"))
            elif isinstance(image, dict) and image.get("path"):
                visuals.append(Image.open(image["path"]).convert("RGB"))
            elif isinstance(image, str):
                visuals.append(Image.open(image).convert("RGB"))
        except Exception as error:
            eval_logger.warning("PhysReason failed to decode an image: {}", error)
    return visuals


def physreason_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Build the prompt from context + sub-questions.

    Formats the physics problem with all sub-questions numbered,
    and asks for step-by-step reasoning with clearly labeled answers.
    """
    context = doc.get("context", "")
    sub_questions = doc.get("sub_questions", [])

    prompt_parts = []
    prompt_parts.append(context.strip())

    if sub_questions:
        prompt_parts.append("")
        for i, sq in enumerate(sub_questions, 1):
            prompt_parts.append(f"({i}) {sq.strip()}")

    prompt_parts.append("")
    prompt_parts.append("Solve each sub-question step by step. For each sub-question, show your reasoning and then give the final answer. Format each final answer as: Answer (N): <your answer>")

    return "\n".join(prompt_parts)


def _normalize_answer(text):
    """Normalize a LaTeX/math answer string for comparison."""
    s = text.strip()
    s = s.strip("$")
    s = re.sub(r"\\(?:text|mathrm|mathsf|mathit)\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[,;:!\s]", "", s)
    s = s.replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_answers_from_response(response, num_expected):
    """Try to extract numbered answers from model response."""
    answers = []

    pattern = r"Answer\s*\(?(\d+)\)?[:\s]+(.+?)(?=Answer\s*\(?\d|$)"
    matches = re.findall(pattern, response, re.IGNORECASE | re.DOTALL)

    if matches:
        matches.sort(key=lambda x: int(x[0]))
        for _, ans in matches:
            ans_clean = ans.strip().split("\n")[0].strip()
            ans_clean = ans_clean.rstrip(".")
            answers.append(ans_clean)

    if len(answers) < num_expected:
        boxed = re.findall(r"\\boxed\{([^}]+)\}", response)
        if len(boxed) >= num_expected:
            answers = boxed[:num_expected]

    return answers


def physreason_process_results(doc, results):
    """Process model output and compare against ground truth answers."""
    prediction = results[0].strip() if results else ""
    answers_gt = doc.get("answers", [])
    num_sq = len(answers_gt)
    difficulty = doc.get("difficulty", "unknown")

    extracted = _extract_answers_from_response(prediction, num_sq)

    correct = 0
    for i, gt in enumerate(answers_gt):
        gt_norm = _normalize_answer(gt)
        if i < len(extracted):
            pred_norm = _normalize_answer(extracted[i])
            if gt_norm == pred_norm:
                correct += 1

    accuracy = correct / max(num_sq, 1)

    eval_result = {
        "problem_id": doc.get("problem_id", ""),
        "difficulty": difficulty,
        "num_sub_questions": num_sq,
        "correct": correct,
        "accuracy": accuracy,
    }

    return {"physreason_accuracy": eval_result}


def physreason_aggregate_results(results):
    """Aggregate per-problem accuracy into overall score."""
    if not results:
        eval_logger.warning("Empty results list for PhysReason. Returning 0.0")
        return 0.0

    accuracies = [r["accuracy"] for r in results]
    overall = float(np.mean(accuracies))

    by_difficulty = {}
    for r in results:
        d = r["difficulty"]
        if d not in by_difficulty:
            by_difficulty[d] = []
        by_difficulty[d].append(r["accuracy"])

    for d in sorted(by_difficulty.keys()):
        acc = float(np.mean(by_difficulty[d]))
        count = len(by_difficulty[d])
        eval_logger.info(f"PhysReason [{d}]: {acc:.4f} ({count} problems)")

    eval_logger.info(f"PhysReason [overall]: {overall:.4f} ({len(results)} problems)")
    return overall
