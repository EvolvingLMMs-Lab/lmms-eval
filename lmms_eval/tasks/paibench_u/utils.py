import os
import re
from pathlib import Path

import yaml
from loguru import logger as eval_logger

hf_home = os.getenv("HF_HOME", "~/.cache/huggingface")
base_cache_dir = os.path.expanduser(hf_home)


with open(Path(__file__).parent / "paibench_u.yaml", "r") as f:
    raw_data_test = f.readlines()
    safe_data_test = []
    for i, line in enumerate(raw_data_test):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data_test.append(line)
cache_name_test = yaml.safe_load("".join(safe_data_test))["dataset_kwargs"]["cache_dir"]
cache_dir_test = os.path.join(base_cache_dir, cache_name_test)


def paibench_u_doc_to_visual(doc):
    video_path = os.path.join(cache_dir_test, "videos", doc["video_path"])
    if os.path.exists(video_path):
        return [video_path]
    else:
        eval_logger.warning(f"Video path: {video_path} does not exist, skipping sample")
        return []


def paibench_u_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    options = doc["index2ans"]
    options_list = []
    for key in sorted(options.keys()):
        value = options[key]
        if value is not None:
            options_list.append(f"{key}. {value}")

    full_string = question + "\n" + "\n".join(options_list)
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    full_prompt = pre_prompt + full_string + post_prompt
    return full_prompt


def parse_multi_choice_response(response):
    """
    Changed from MMMU-style complex parsing into simple parsing.
    Fixed to avoid 'D. A book' be parsed as A.
    Same as original LongVideoBench paper (from author Haoning Wu), if parsing failed,
    it will assign a random choice to model.
    """
    import random

    s = response.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCDE]", s):
        return random.choice(["A", "B", "C", "D", "E"])

    # Try multiple patterns for robustness: (A), [A], A), A
    patterns = [
        r"\(([ABCDE])\)",  # (A)
        r"\[([ABCDE])\]",  # [A]
        r"([ABCDE])\)",  # A)
        r"([ABCDE])\.",  # A.
        r"([ABCDE])",  # A
    ]

    for pattern in patterns:
        matches = re.search(pattern, s)
        if matches is not None:
            return matches[1]

    return random.choice(["A", "B", "C", "D", "E"])


def paibench_u_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name, value: metric value
    """
    pred = results[0]

    pred_ans = parse_multi_choice_response(pred)

    category = doc["category"]
    subcategory = doc["subcategory"]
    data_dict = {"question_id": doc["question"], "pred_answer": pred_ans, "answer": doc["answer"], "category": category, "subcategory": subcategory}

    return {"paibench_u_perception_score": data_dict}


def paibench_u_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A dictionary containing all metrics including overall accuracy,
        category accuracies, and subcategory accuracies
    """
    # Initialize dictionaries for categories and subcategories
    total_correct = 0
    total_answered = 0
    category_scores = {}
    subcategory_scores = {}

    for result in results:
        category = result["category"]
        subcategory = result["subcategory"].split(":")[0]

        # Initialize category scores if not exists
        if category not in category_scores:
            category_scores[category] = {"correct": 0, "answered": 0}

        # Initialize subcategory scores if not exists
        if subcategory not in subcategory_scores:
            subcategory_scores[subcategory] = {"correct": 0, "answered": 0}

        # Update counts
        is_correct = result["pred_answer"] == result["answer"]

        total_answered += 1
        total_correct += is_correct
        category_scores[category]["answered"] += 1
        category_scores[category]["correct"] += is_correct
        subcategory_scores[subcategory]["answered"] += 1
        subcategory_scores[subcategory]["correct"] += is_correct

    # Calculate overall accuracy
    overall_accuracy = 100 * total_correct / total_answered if total_answered > 0 else 0

    # Build the metrics dictionary
    metrics = {"overall": overall_accuracy, "category": {}, "subcategory": {}}

    # Log overall accuracy
    eval_logger.info("=== Overall Accuracy ===")
    eval_logger.info(f"Overall: {overall_accuracy:.1f}% ({total_correct}/{total_answered})")

    # Calculate and add accuracy for each category
    eval_logger.info("=== Category Accuracy ===")
    for category, scores in category_scores.items():
        accuracy = 100 * scores["correct"] / scores["answered"] if scores["answered"] > 0 else 0
        eval_logger.info(f"Category: {category}: {accuracy:.1f}% ({scores['correct']}/{scores['answered']})")
        metrics["category"][category] = accuracy

    # Calculate and add accuracy for each subcategory
    eval_logger.info("=== Subcategory Accuracy ===")
    for subcategory, scores in subcategory_scores.items():
        accuracy = 100 * scores["correct"] / scores["answered"] if scores["answered"] > 0 else 0
        eval_logger.info(f"Subcategory: {subcategory}: {accuracy:.1f}% ({scores['correct']}/{scores['answered']})")
        metrics["subcategory"][subcategory] = accuracy

    return metrics["overall"]
