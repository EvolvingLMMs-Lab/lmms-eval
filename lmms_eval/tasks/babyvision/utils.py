"""
BabyVision Utils
MLLM evaluation task using LLM-as-Judge API
"""

import os
import re
from collections import defaultdict
from typing import Optional

from loguru import logger as eval_logger

from lmms_eval.tasks.babyvision.prompt import build_judge_prompt


def _get_openai_client() -> Optional["OpenAI"]:
    """Get OpenAI client instance."""
    api_key = os.getenv("BABYVISION_API_KEY")
    base_url = os.getenv("BABYVISION_BASE_URL")

    if not api_key:
        eval_logger.error("API key not found. Set BABYVISION_API_KEY environment variable.")
        return None

    from openai import OpenAI

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    return OpenAI(**client_kwargs)


def parse_bool_response(response_text: str) -> Optional[bool]:
    """Parse boolean response from LLM."""
    text = response_text.strip().lower()
    if "true" in text:
        return True
    elif "false" in text:
        return False
    else:
        eval_logger.warning(f"Could not parse boolean from response: {response_text[:200]}...")
        return None


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract the content from \\boxed{} pattern.
    Handles MiMo-VL's <think>...</think> reasoning format.
    """
    if not text:
        return None

    # For models with <think>...</think> format, prioritize content after </think>
    think_end = text.find("</think>")
    if think_end != -1:
        text_after_think = text[think_end + 8 :]
        # Try to find \boxed{} in the content after </think>
        pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
        matches = re.findall(pattern, text_after_think)
        if matches:
            return matches[-1].strip()

    # Standard \boxed{} extraction from full text
    pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()

    # Fallback: try to find answer patterns
    # Pattern: "answer is X" or "Answer: X"
    answer_patterns = [
        r"(?:answer|Answer|ANSWER)\s*(?:is|:)\s*[\"']?([A-Za-z0-9,\s\(\)\-]+)[\"']?",
        r"(?:final answer|Final Answer)\s*(?:is|:)\s*[\"']?([A-Za-z0-9,\s\(\)\-]+)[\"']?",
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()

    return None


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if not answer:
        return ""
    # Remove extra whitespace, convert to lowercase
    normalized = " ".join(answer.lower().split())
    # Remove common punctuation
    normalized = re.sub(r"[.,;:!?]$", "", normalized)
    return normalized


def babyvision_doc_to_visual(doc):
    """Extract input image from document."""
    image = doc["image"]
    return [image.convert("RGB")]


def babyvision_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Build question text from document."""
    question = doc["question"]
    ans_type = doc["ansType"]

    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    # Add choices for multiple-choice questions (aligned with BabyVision repo format)
    if ans_type == "choice" and doc.get("options"):
        options = doc["options"]
        # Format: (A) option, (B) option, etc. with "Choices:" label
        options_str = "\n".join([f"({chr(ord('A') + i)}) {opt}" for i, opt in enumerate(options)])
        return f"{pre_prompt}{question}\nChoices:\n{options_str}{post_prompt}"

    return f"{pre_prompt}{question}{post_prompt}"


def babyvision_doc_to_target(doc):
    """Get ground truth answer from document."""
    ans_type = doc["ansType"]

    if ans_type == "choice":
        choice_ans = doc.get("choiceAns")
        if choice_ans is not None:
            return chr(ord("A") + int(choice_ans))
    else:
        blank_ans = doc.get("blankAns")
        if blank_ans is not None:
            return str(blank_ans)

    return ""


def babyvision_process_results(doc, results, **kwargs):
    """Process model predictions and evaluate using LLM-as-Judge API."""
    task_type = doc["type"]
    subtype = doc["subtype"]

    # Get ground truth
    groundtruth = babyvision_doc_to_target(doc)
    question = doc["question"]

    # Get model response
    model_response = results[0] if results else ""

    # Extract answer from model response
    parsed_pred = extract_boxed_answer(model_response)
    if parsed_pred is None:
        parsed_pred = model_response[:500] if model_response else ""

    # Call LLM judge
    client = _get_openai_client()
    if client is None:
        return {
            "babyvision_overall_accuracy": {
                "task_type": task_type,
                "subtype": subtype,
                "correct": False,
                "error": "No API client",
            }
        }

    try:
        prompt = build_judge_prompt(question, groundtruth, parsed_pred)
        model_name = os.getenv("BABYVISION_MODEL_NAME", "gpt-4o")

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        )

        judge_response = response.choices[0].message.content
        correct = parse_bool_response(judge_response)

        if correct is None:
            correct = False

    except Exception as e:
        eval_logger.error(f"LLM judge API call failed: {e}")
        return {
            "babyvision_overall_accuracy": {
                "task_type": task_type,
                "subtype": subtype,
                "correct": False,
                "error": str(e),
            }
        }

    return {
        "babyvision_overall_accuracy": {
            "task_type": task_type,
            "subtype": subtype,
            "correct": correct,
            "parsed_pred": parsed_pred,
            "groundtruth": groundtruth,
        }
    }


def babyvision_aggregate_results(results):
    """Aggregate results across all samples."""
    if not results:
        return 0.0

    correct_count = sum(1 for r in results if r.get("correct", False))
    total_count = len(results)

    accuracy = correct_count / total_count if total_count > 0 else 0.0

    # Count by task type/subtype
    task_type_counts = defaultdict(lambda: {"correct": 0, "total": 0})
    error_count = 0

    for r in results:
        task_type = r.get("task_type", "unknown")
        subtype = r.get("subtype", "unknown")
        key = f"{task_type}/{subtype}"
        task_type_counts[key]["total"] += 1
        if r.get("correct", False):
            task_type_counts[key]["correct"] += 1
        if r.get("error"):
            error_count += 1

    # Log results
    eval_logger.info("=" * 60)
    eval_logger.info("BabyVision Final Results")
    eval_logger.info("=" * 60)
    eval_logger.info(f"Overall Accuracy: {accuracy:.4f} ({correct_count}/{total_count})")
    if error_count > 0:
        eval_logger.warning(f"Errors encountered: {error_count}/{total_count}")
    eval_logger.info("-" * 60)
    eval_logger.info("Accuracy by Task Type/Subtype:")
    for key in sorted(task_type_counts.keys()):
        stats = task_type_counts[key]
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        eval_logger.info(f"  {key}: {acc:.4f} ({stats['correct']}/{stats['total']})")
    eval_logger.info("=" * 60)

    return float(accuracy)
