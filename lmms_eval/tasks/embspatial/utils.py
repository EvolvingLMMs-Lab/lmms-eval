import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

eval_logger = logging.getLogger("lmms-eval")

with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))uv run pytest


def _extract_answer_letter(text: str) -> str:
    """
    Extract the answer choice letter from a string.

    Examples:
    'A answer1' -> 'A'
    'A) answer2' -> 'A'
    '(B) answer' -> 'B'
    'C' -> 'C'
    '(C)' -> 'C'
    'A.' -> 'A'

    Return an empty string if no letter is found.
    """
    text = text.strip()
    match = re.match(r'[\(\s]*([A-Z])[\)\.\s]*', text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return ""


def embspatial_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs: Optional[dict[str, Any]] = None) -> str:
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    options = doc["answer_options"]
    formatted_lines = []
    for i, item in enumerate(options):
        letter = chr(65 + i)
        formatted_lines.append(f"{letter}) {item}")
    options_string = "\n".join(formatted_lines)
    prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") + doc["question"] + "\n" + options_string
    return prompt


def embspatial_doc_to_visual(doc: dict) -> list:
    return [doc["image"].convert("RGB")]


def embspatial_process_results(doc, results):
    choices = ["A", "B", "C", "D"]
    key_name = "embspatial_acc"
    # extract grounded answer
    grounded_output = choices[doc["answer"]]
    response = results[0]

    # extract predicted answer
    pred_letter = _extract_answer_letter(response)
    flag = pred_letter == grounded_output

    omnispatial_submission = {"id": doc["question_id"], "gt_content": grounded_output, "pred": response, "sub_task": doc["relation"], "is_correct": flag}
    return {key_name: omnispatial_submission}


def embspatial_aggregate_results(results: List[Dict]):
    sub_task_to_eval_samples = defaultdict(list)
    total_samples = len(results)
    total_correct = 0

    for sample in results:
        sub_task = sample["sub_task"]
        is_correct = sample["is_correct"]

        if is_correct:
            total_correct += 1
            sub_task_to_eval_samples[sub_task].append(1)
        else:
            sub_task_to_eval_samples[sub_task].append(0)

    accuracy = total_correct / total_samples if total_samples > 0 else 0
    sub_task_accuracies = {sub_task: sum(scores) / len(scores) for sub_task, scores in sub_task_to_eval_samples.items()}

    eval_logger.info("%-40s", 'EmbSpatial Per-Sub-Task Accuracy')
    eval_logger.info("-" * 40)

    for sub_task, acc in sub_task_accuracies.items():
        eval_logger.info("%-20s: %.4f", sub_task, acc)

    eval_logger.info("=" * 40)
    return accuracy
