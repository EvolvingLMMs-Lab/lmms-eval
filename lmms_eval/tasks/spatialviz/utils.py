import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import yaml
from huggingface_hub import snapshot_download
from loguru import logger as eval_logger
from PIL import Image

with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)
    config = yaml.safe_load("".join(safe_data))

cache_dir = snapshot_download(
    repo_id=config["dataset_path"],
    repo_type="dataset",
    local_dir_use_symlinks=False,
)


def spatialviz_doc_to_visual(doc: Dict[str, Any]) -> List[Image.Image]:
    visual = []

    category = doc["Category"]
    task = doc["Task"]
    level = doc["Level"]
    image_id = doc["Image_id"]
    image_path = f"{cache_dir}/{category}/{task}/{level}/{image_id}.png"

    if os.path.exists(image_path):
        visual.append(Image.open(image_path).convert("RGB"))
    else:
        raise FileNotFoundError(f"image path: {image_path} does not exist.")
    return visual


def spatialviz_doc_to_text(doc: Dict[str, Any]) -> str:
    ops = ["A", "B", "C", "D"]
    prompt = (
        "You should first provide a reasoning process, then provide a single "
        "option(A, B, C or D) as the final answer. The reasoning process and "
        "the answer are enclosed within <think></think> and <answer></answer> "
        "tags, respectively, i.e., <think>reasoning process</think>, "
        "<answer>answer</answer>.\n"
    )
    question = doc["Question"]
    choices = doc["Choices"]
    choice_text = ""

    for i, choice in enumerate(choices):
        choice_text += ops[i] + ". " + choice + "\n"
    text = prompt + "Question: " + question + "\n" + choice_text
    return text


def spatialviz_process_results(
    doc: Dict[str, Any], results: List[str]
) -> Dict[str, Dict[str, Any]]:
    key_name = "spatialviz_score"
    grounded_output = doc["Answer"]
    response = results[0]

    think_pattern = r"<think>(.*?)</think>"
    answer_pattern = r"<answer>(.*?)</answer>"

    think_match = re.search(think_pattern, response, re.DOTALL)
    answer_match = re.search(answer_pattern, response, re.DOTALL)

    op: List[str] = []
    if think_match and answer_match:
        final_answer = answer_match.group(1).strip()
        pred_answer = final_answer.split(".")[0]
        op = re.findall(r"[A-D]", pred_answer)
    else:
        eval_logger.debug("No match for think/answer tags in response")
        final_answer_patterns = [
            "<answer>",
            "Answer:",
            "Final answer",
            "final answer",
            "Final Answer",
            "the answer is",
            "The answer is",
            "correct answer",
            "Correct answer",
            "Correct Answer",
            "答案",
            "correct path",
        ]
        if len(response) == 1:
            op = re.findall(r"[A-D]", response)
        else:
            for pattern in final_answer_patterns:
                if pattern in response:
                    response = response.split(pattern)[-1].strip()
                    op = re.findall(r"[A-D]", response.split(".")[0])
                    break

    op = list(set(op))

    if len(op) == 1 and grounded_output == op[0].upper():
        is_correct = True
    else:
        is_correct = False

    query = spatialviz_doc_to_text(doc)
    spatialviz_submission = {
        "id": doc["Image_id"],
        "query": query,
        "gt_content": grounded_output,
        "pred": response,
        "category": doc["Category"],
        "task": doc["Task"],
        "level": doc["Level"],
        "is_correct": is_correct,
    }
    return {key_name: spatialviz_submission}


def spatialviz_aggregate_results(results: List[Dict[str, Any]]) -> float:
    task_to_eval_samples: Dict[str, List[int]] = defaultdict(list)
    category_to_eval_samples: Dict[str, List[int]] = defaultdict(list)
    key_to_eval_samples: Dict[str, List[int]] = defaultdict(list)
    total_samples = len(results)
    total_correct = 0

    for sample in results:
        task = sample["task"]
        category = sample["category"]
        level = sample["level"]
        key = f"{category}-{task}-{level}"
        is_correct = sample["is_correct"]

        if is_correct:
            total_correct += 1
            task_to_eval_samples[task].append(1)
            category_to_eval_samples[category].append(1)
            key_to_eval_samples[key].append(1)
        else:
            task_to_eval_samples[task].append(0)
            category_to_eval_samples[category].append(0)
            key_to_eval_samples[key].append(0)

    accuracy = total_correct / total_samples if total_samples > 0 else 0
    task_accuracies = {
        task: sum(scores) / len(scores) for task, scores in task_to_eval_samples.items()
    }
    category_accuracies = {
        category: sum(scores) / len(scores)
        for category, scores in category_to_eval_samples.items()
    }
    key_accuracies = {
        key: sum(scores) / len(scores) for key, scores in key_to_eval_samples.items()
    }

    eval_logger.info(f"{'Total Samples':<20}: {total_samples}")
    eval_logger.info(f"{'Total Correct':<20}: {total_correct}")
    eval_logger.info(f"{'Overall Accuracy':<20}: {accuracy:.4f}")

    eval_logger.info(f"{'Per-Task Accuracy':<40}")
    eval_logger.info("-" * 40)
    for task, acc in task_accuracies.items():
        eval_logger.info(f"{task:<20}: {acc:.4f}")

    eval_logger.info(f"{'Per-Category Accuracy':<40}")
    eval_logger.info("-" * 40)
    for category, acc in category_accuracies.items():
        eval_logger.info(f"{category:<20}: {acc:.4f}")
    eval_logger.info("=" * 40)

    eval_logger.info(f"{'Per-Key Accuracy':<40}")
    eval_logger.info("-" * 40)
    for key, acc in key_accuracies.items():
        eval_logger.info(f"{key:<20}: {acc:.4f}")

    return accuracy
