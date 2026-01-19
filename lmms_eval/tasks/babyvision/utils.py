"""
BabyVision Benchmark Utilities

BabyVision is a benchmark for evaluating visual reasoning capabilities of MLLMs
on tasks that even 3-year-old children can solve, but remain challenging for AI.

Dataset: UnipatAI/BabyVision (388 items)
Categories: Fine-grained Discrimination, Visual Tracking, Spatial Perception,
            Visual Pattern Recognition

Reference: https://github.com/UniPat-AI/BabyVision
"""

import os
import re
from collections import defaultdict

from loguru import logger as eval_logger

from lmms_eval.llm_judge import ServerConfig, get_server

# LLM Judge configuration for blank question evaluation
API_TYPE = os.getenv("API_TYPE", "openai")
MODEL_VERSION = os.getenv("MODEL_VERSION", "gpt-4o-2024-11-20")

# Lazy-loaded judge server
_server = None


def get_judge_server():
    """Lazily initialize the judge server."""
    global _server
    if _server is None:
        server_config = ServerConfig(model_name=MODEL_VERSION)
        _server = get_server(server_name=API_TYPE, config=server_config)
    return _server


# Type ID mapping for consistent subtype ordering (matches original BabyVision)
TYPE_ID_MAP = {
    "Fine-grained Discrimination": "1",
    "Visual Tracking": "2",
    "Spatial Perception": "3",
    "Visual Pattern Recognition": "4",
}

BABYVISION_CATEGORIES = {
    "Fine-grained Discrimination": [
        "2D Pattern Completion",
        "Count Clusters",
        "Count Same Patterns",
        "Find the different",
        "Find the same",
        "Find the shadow",
        "Pattern and Color Completion",
        "Reconstruction",
    ],
    "Visual Tracking": [
        "Connect the lines",
        "Lines Observation",
        "Maze",
        "Metro map",
        "Recognize numbers and letters",
    ],
    "Spatial Perception": [
        "3D Cube Unfold",
        "3D Pattern Completion",
        "3D Views",
        "Count 3D blocks",
        "Paper Folding",
    ],
    "Visual Pattern Recognition": [
        "Logic Patterns",
        "Mirroring Patterns",
        "Overlay Patterns",
        "Rotation Patterns",
    ],
}


def babyvision_doc_to_visual(doc):
    """Extract visual from document."""
    return [doc["image"].convert("RGB")]


def babyvision_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Convert document to text prompt."""
    question = doc["question"]
    ans_type = doc["ansType"]

    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    if ans_type == "choice":
        options = doc["options"]
        if options:
            options_str = "\n".join(
                [f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(options)]
            )
            return f"{pre_prompt}{question}\n{options_str}{post_prompt}"
    return f"{pre_prompt}{question}{post_prompt}"


def babyvision_doc_to_target(doc):
    """Get the target answer from document."""
    ans_type = doc["ansType"]

    if ans_type == "choice":
        choice_ans = doc["choiceAns"]
        if choice_ans is not None:
            return chr(ord("A") + int(choice_ans))
        return ""
    else:
        return doc["blankAns"] if doc["blankAns"] is not None else ""


def extract_boxed_answer(text):
    """
    Extract content from \\boxed{} pattern (matches original BabyVision eval).
    Also supports: <|begin_of_box|>...<|end_of_box|>
    """
    if not text:
        return None

    # Match \boxed{...} with nested braces support
    pattern = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()

    # Alternative pattern
    pattern_alt = r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>"
    matches_alt = re.findall(pattern_alt, text, re.DOTALL)
    if matches_alt:
        return matches_alt[-1].strip()

    return None


def parse_choice_response(response, num_choices):
    response = response.strip()
    all_choices = [chr(ord("A") + i) for i in range(num_choices)]

    # Try to extract from boxed answer first
    boxed = extract_boxed_answer(response)
    if boxed:
        boxed_upper = boxed.upper().strip()
        if len(boxed_upper) == 1 and boxed_upper in all_choices:
            return boxed_upper

    response_upper = response.upper()

    for choice in all_choices:
        if f"({choice})" in response_upper:
            return choice

    for choice in all_choices:
        if f"{choice}." in response_upper:
            return choice

    for choice in all_choices:
        if f" {choice} " in f" {response_upper} ":
            return choice

    if len(response) == 1 and response.upper() in all_choices:
        return response.upper()

    for char in response_upper:
        if char in all_choices:
            return char

    return response


def normalize_answer(answer):
    if not answer:
        return ""
    answer = " ".join(answer.strip().lower().split())
    answer = re.sub(r"[,;:\.\!\?]+$", "", answer)
    return answer


def evaluate_with_llm_judge(question, gt_answer, pred_answer):
    """
    Use LLM judge to evaluate blank question answers.
    Matches original BabyVision evaluation strategy.
    """
    try:
        server = get_judge_server()
        result = server.evaluate_binary(
            question=question,
            answer=str(gt_answer),
            prediction=str(pred_answer),
            output_format="0/1",
        )

        if result["success"]:
            judge_response = result["result"]
            is_correct = (
                int(judge_response) == 1
                if isinstance(judge_response, str)
                else judge_response == 1
            )
            return is_correct, "llm_judge"
        else:
            eval_logger.warning(
                f"Judge evaluation failed: {result.get('raw_response')}"
            )
            return False, "llm_judge_failed"
    except Exception as e:
        eval_logger.error(f"Error in LLM judge: {e}")
        return False, "llm_judge_error"


def babyvision_process_results(doc, results):
    pred = results[0] if results else ""
    ans_type = doc["ansType"]
    task_type = doc["type"]
    subtype = doc["subtype"]
    task_id = doc["taskId"]

    if ans_type == "choice":
        num_choices = len(doc["options"]) if doc["options"] else 4
        parsed_pred = parse_choice_response(pred, num_choices)
        target = babyvision_doc_to_target(doc)
        correct = parsed_pred.upper() == target.upper()
        eval_method = "exact_match"
    else:
        target = babyvision_doc_to_target(doc)

        boxed_answer = extract_boxed_answer(pred)
        parsed_pred = boxed_answer if boxed_answer else pred.strip()

        norm_pred = normalize_answer(parsed_pred)
        norm_target = normalize_answer(target)

        if (
            norm_pred == norm_target
            or norm_target in norm_pred
            or norm_pred in norm_target
        ):
            correct = True
            eval_method = "exact_match"
        else:
            correct, eval_method = evaluate_with_llm_judge(
                doc["question"], target, parsed_pred
            )

    return {
        "babyvision_acc": {
            "task_id": task_id,
            "type": task_type,
            "subtype": subtype,
            "ans_type": ans_type,
            "target": target,
            "parsed_pred": parsed_pred,
            "correct": correct,
            "eval_method": eval_method,
        }
    }


def babyvision_aggregate_results(results):
    total_correct = sum(1 for r in results if r["correct"])
    total_count = len(results)
    overall_acc = total_correct / total_count if total_count > 0 else 0

    type_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    subtype_stats = defaultdict(lambda: {"total": 0, "correct": 0})

    for r in results:
        task_type = r["type"]
        subtype = r["subtype"]
        type_id = TYPE_ID_MAP.get(task_type, "0")

        type_stats[task_type]["total"] += 1
        subtype_key = f"{type_id}{task_type}/{subtype}"
        subtype_stats[subtype_key]["total"] += 1

        if r["correct"]:
            type_stats[task_type]["correct"] += 1
            subtype_stats[subtype_key]["correct"] += 1

    eval_logger.info("=" * 60)
    eval_logger.info("BabyVision Results")
    eval_logger.info("=" * 60)
    eval_logger.info(
        f"Overall Accuracy: {overall_acc:.4f} ({total_correct}/{total_count})"
    )
    eval_logger.info("")
    eval_logger.info("Type-wise Accuracy:")
    for task_type in sorted(type_stats.keys()):
        stats = type_stats[task_type]
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        eval_logger.info(
            f"  {task_type}: {acc:.4f} ({stats['correct']}/{stats['total']})"
        )

    eval_logger.info("")
    eval_logger.info("Subtype-wise Accuracy:")
    for subtype_key in sorted(subtype_stats.keys()):
        stats = subtype_stats[subtype_key]
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        eval_logger.info(
            f"  {subtype_key}: {acc:.4f} ({stats['correct']}/{stats['total']})"
        )
    eval_logger.info("=" * 60)

    return overall_acc
