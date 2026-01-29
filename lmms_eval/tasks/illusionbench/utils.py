import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from loguru import logger as eval_logger

with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
    config = yaml.safe_load("".join(safe_data))


def illusionbench_doc_to_visual(doc: dict[str, Any]) -> list:
    """Extract visual from document for lmms-eval framework."""
    return [doc["image"].convert("RGB")]


def illusionbench_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs: dict[str, Any] | None = None) -> str:
    """Format question text based on question type for lmms-eval framework."""
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    question = doc["question"]
    question_type = doc["question_type"]

    if question_type == "TF":
        pre_prompt = lmms_eval_specific_kwargs.get("tf_pre_prompt", "")
        post_prompt = lmms_eval_specific_kwargs.get("tf_post_prompt", "\nAnswer with True or False only.")
    else:
        pre_prompt = lmms_eval_specific_kwargs.get("mc_pre_prompt", "")
        post_prompt = lmms_eval_specific_kwargs.get(
            "mc_post_prompt",
            "\nAnswer with the option number (i, ii, iii, or iv) only.",
        )

    return f"{pre_prompt}{question}{post_prompt}"


def _parse_tf_answer(response: str) -> str:
    """Parse True/False answer from response."""
    response = response.lower().strip()

    if "true" in response[:15]:
        return "true"
    elif "false" in response[:15]:
        return "false"
    elif response.startswith("t"):
        return "true"
    elif response.startswith("f"):
        return "false"
    elif "yes" in response[:15]:
        return "true"
    elif "no" in response[:15]:
        return "false"
    return "unknown"


def _parse_mc_answer(response: str) -> str:
    """Parse multiple choice answer from model response."""
    response = response.strip().lower()

    patterns = [
        r"^\s*\(?([ivx]+)\)?[\.\)\s:]",
        r"^\s*([ivx]+)[\.\)\s:]",
        r"\b([ivx]+)\b",
        r"^\s*\(?\s*([ivx]+)\s*\)?",
    ]

    for pattern in patterns:
        match = re.search(pattern, response[:50])
        if match:
            return match.group(1)

    return response[:20] if response else ""


def _normalize_mc_answer(answer: str) -> str:
    """Normalize multiple choice answer for comparison."""
    answer = answer.strip().lower()
    match = re.search(r"([ivx]+)", answer)
    if match:
        return match.group(1)
    return answer


def illusionbench_process_results(doc: dict[str, Any], results: list[str]) -> dict[str, dict]:
    """Process results and compute correctness for lmms-eval framework."""
    pred = results[0]
    question_type = doc["question_type"]
    correct_answer = doc["answer"].strip()
    category = doc.get("category", "unknown")

    if question_type == "TF":
        parsed_pred = _parse_tf_answer(pred)
        correct_normalized = correct_answer.lower().strip()
        is_correct = parsed_pred == correct_normalized
    else:
        parsed_pred = _parse_mc_answer(pred)
        correct_normalized = _normalize_mc_answer(correct_answer)
        is_correct = parsed_pred == correct_normalized

    # Use image_name as question_id if question_id not present
    question_id = doc.get("question_id", doc.get("image_name", "unknown"))

    return {
        "illusionbench_acc": {
            "question_id": question_id,
            "category": category,
            "question_type": question_type,
            "correct_answer": correct_answer,
            "parsed_pred": parsed_pred,
            "raw_pred": pred,
            "is_correct": is_correct,
        }
    }


def illusionbench_aggregate_results(results: list[dict]) -> float:
    """Aggregate results to compute overall accuracy for lmms-eval framework."""
    category_results: dict[str, dict[str, list]] = defaultdict(lambda: {"correct": [], "total": []})
    type_results: dict[str, dict[str, list]] = defaultdict(lambda: {"correct": [], "total": []})

    total_correct = 0
    total_count = 0

    for result in results:
        category = result["category"]
        question_type = result["question_type"]
        is_correct = result["is_correct"]

        category_results[category]["total"].append(1)
        type_results[question_type]["total"].append(1)
        total_count += 1

        if is_correct:
            category_results[category]["correct"].append(1)
            type_results[question_type]["correct"].append(1)
            total_correct += 1

    for category, data in sorted(category_results.items()):
        acc = sum(data["correct"]) / len(data["total"]) * 100
        eval_logger.info(f"Category {category}: {acc:.2f}%")

    for qtype, data in sorted(type_results.items()):
        acc = sum(data["correct"]) / len(data["total"]) * 100
        type_name = "True/False" if qtype == "TF" else "Multiple Choice"
        eval_logger.info(f"{type_name}: {acc:.2f}%")

    overall_acc = total_correct / total_count * 100 if total_count > 0 else 0
    eval_logger.info(f"Overall: {overall_acc:.2f}%")

    return overall_acc
