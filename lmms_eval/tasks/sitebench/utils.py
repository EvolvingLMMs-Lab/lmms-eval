# taken directly from https://github.com/wenqi-wang20/SITE-Bench/blob/main/eval_scripts/sitebench/utils.py
import os
import random
import string
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

UpperLetters = list(string.ascii_uppercase)
Categories = {
    "counting & existence",
    "spatial relationship reasoning",
    "object localization & positioning",
    "3d information understanding",
    "movement prediction & navigation",
    "multi-view & cross-image reasoning",
}

# Mapping from category name to metric key suffix
CATEGORY_TO_METRIC_KEY = {
    "3d information understanding": "3d_information_understanding",
    "counting & existence": "counting_and_existence",
    "movement prediction & navigation": "movement_prediction_and_navigation",
    "multi-view & cross-image reasoning": "multiview_and_crossimage_reasoning",
    "object localization & positioning": "object_localization_and_positioning",
    "spatial relationship reasoning": "spatial_relationship_reasoning",
}

# Get the cache directory from the config file
hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
# cache_dir = os.path.join(hf_home, cache_dir)
# base_cache_dir = config["dataset_kwargs"]["cache_dir"]
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "site_image.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]
cache_dir = os.path.join(base_cache_dir, cache_name)


##################
# Helper functions adapted from MMMU's utils.py.
##################
def parse_multi_choice_response(response, all_choices):
    """
    Parse the prediction from the generated response.
    Return the predicted choice letter e.g., A, B, C, D.
    """
    # # Clean response of unwanted characters
    # for char in [",", ".", "!", "?", ";", ":", "'"]:
    #     response = response.strip(char)
    response = " " + response + " "  # Add space to avoid partial match

    candidates = []
    # Look for choices with parentheses, e.g., (A)
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)

    # Look for simple choices, e.g., A, B, C
    if len(candidates) == 0:
        for choice in all_choices:
            if f" {choice} " in response:
                candidates.append(choice)

    # Look for choices with periods, e.g., A., B., C.
    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice}." in response:
                candidates.append(choice)

    # Look for choices with periods, e.g., A:, B:, C:.
    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice}:" in response or f":{choice}" in response or f": {choice}" in response:
                candidates.append(choice)

    # If no candidates, randomly choose one
    if len(candidates) == 0:
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        # If more than one candidate, choose the last one found
        start_indexes = [response.rfind(f" {can} ") for can in candidates]
        pred_index = candidates[np.argmax(start_indexes)]
    else:
        # If only one candidate, use it
        pred_index = candidates[0]

    return pred_index


def spatial_doc_to_visual_image(doc):
    imgs = []
    for image_path in doc["visual"]:
        full_image_path = os.path.join(cache_dir, image_path)
        imgs.append(Image.open(full_image_path).convert("RGB"))
    return imgs


def spatial_doc_to_visual_video(doc):
    return [os.path.join(cache_dir, doc["visual"][0])]


def spatial_doc_to_text_image(doc, lmmseval_specific_kwargs=None):
    question = doc["question"].strip()
    options = doc["options"]
    option_text = "\n".join(f"{UpperLetters[i]}: {options[i]}" for i in range(len(options)))

    prompt = ""
    # check if '<image>' is in the question, interleaved format
    if "<image>" not in question and "<image>" not in option_text:
        prompt += "<image>" * len(doc["visual"]) + "\n"

    prompt += "Question: " + question + "\n"
    prompt += "Options:\n" + option_text + "\n"

    # Append post prompt if provided
    if lmmseval_specific_kwargs:
        prompt += lmmseval_specific_kwargs.get("default", {}).get("post_prompt", "")

    return prompt


def spatial_doc_to_text_video(doc, lmmseval_specific_kwargs=None):
    pre_prompt = "Select the best answer to the following multiple-choice question based on the video. Respond with only the letter of the correct option."

    question = doc["question"].strip()
    options = doc["options"]
    option_text = "\n".join(f"{UpperLetters[i]}: {options[i]}" for i in range(len(options)))

    prompt = pre_prompt + "\n"

    # check the pre_prompt
    if lmmseval_specific_kwargs:
        prompt += lmmseval_specific_kwargs.get("default", {}).get("pre_prompt", "")

    prompt += "Question: " + question + "\n"
    prompt += "Options:\n" + option_text + "\n"

    # Append post prompt if provided
    if lmmseval_specific_kwargs:
        prompt += lmmseval_specific_kwargs.get("default", {}).get("post_prompt", "")

    return prompt


def spatial_doc_to_messages_image(doc, lmms_eval_specific_kwargs=None):
    """
    Convert a sitebench image document to chat messages format.
    Builds interleaved image-text messages for chat-based models.

    lmms_eval_specific_kwargs: dict, optional
        A dictionary containing evaluation-specific keyword arguments.
        If 'interleave_visuals' is set to False in the 'default' section,
        the function will generate non-interleaved messages.
    """
    if lmms_eval_specific_kwargs and lmms_eval_specific_kwargs.get("default", {}).get("interleave_visuals", True) is False:
        # Fallback to non-interleaved format - content must be a list for ChatMessages
        question = spatial_doc_to_text_image(doc, lmms_eval_specific_kwargs)
        visuals = spatial_doc_to_visual_image(doc)
        # Build content as a list with images first, then text
        content = []
        for visual in visuals:
            content.append({"type": "image", "url": visual})
        content.append({"type": "text", "text": question})
        messages = [{"role": "user", "content": content}]
        eval_logger.debug(f"[sitebench image] Generated messages (non-interleaved): {messages}")
        return messages

    question = spatial_doc_to_text_image(doc, lmms_eval_specific_kwargs)
    visuals = spatial_doc_to_visual_image(doc)

    messages = [{"role": "user", "content": []}]
    interleaved_content = question.split("<image>")

    # Allow more visuals than placeholders by only attaching pre-image text
    # if a corresponding segment exists. Always append the final trailing text.
    for i in range(len(visuals)):
        if i < len(interleaved_content) - 1:
            text = interleaved_content[i].strip()
            if text != "":
                messages[0]["content"].append({"type": "text", "text": text})
        messages[0]["content"].append({"type": "image", "url": visuals[i]})

    # Append the trailing text after the last image
    if len(interleaved_content) > 0:
        trailing_text = interleaved_content[-1].strip()
        if trailing_text:
            messages[0]["content"].append({"type": "text", "text": trailing_text})

    return messages


def spatial_doc_to_messages_video(doc, lmms_eval_specific_kwargs=None):
    """
    Convert a sitebench video document to chat messages format.
    Builds video-text messages for chat-based models.
    """
    question = spatial_doc_to_text_video(doc, lmms_eval_specific_kwargs)
    visuals = spatial_doc_to_visual_video(doc)

    # Video uses a simpler format - video first, then the question text
    messages = [{"role": "user", "content": []}]

    # Add video(s)
    for video_path in visuals:
        messages[0]["content"].append({"type": "video", "url": video_path})

    # Add the question text
    messages[0]["content"].append({"type": "text", "text": question})

    return messages


def spatial_process_results(doc, results):
    response = results[0].strip()
    all_choices = UpperLetters[: len(doc["options"])]
    pred_index = parse_multi_choice_response(response, all_choices)
    gt_index = doc["answer"]
    score = 1.0 if pred_index == gt_index else 0.0

    category = doc["category"]
    dataset = doc["dataset"]
    accuracy_dict = {"overall": score, category: score, dataset: score, "total": 1}

    adjusted_score = score - 1.0 / len(all_choices)
    chance_adjusted_accuracy_dict = {
        "overall": adjusted_score,
        category: adjusted_score,
        dataset: adjusted_score,
        "total": 1.0 - 1.0 / len(all_choices),
    }

    result = {
        "accuracy": accuracy_dict,
        "chance_adjusted_acc": chance_adjusted_accuracy_dict,
    }

    # Per-category accuracy and chance-adjusted accuracy
    for cat_name, metric_key in CATEGORY_TO_METRIC_KEY.items():
        result[f"{metric_key}_acc"] = {"score": score, "category": category, "target_category": cat_name}
        result[f"{metric_key}_caa"] = {"score": adjusted_score, "category": category, "target_category": cat_name, "total": 1.0 - 1.0 / len(all_choices)}

    return result


def spatial_aggregate_results(results):
    total_correct, total_examples = 0, 0
    category_correct, category_total = defaultdict(int), defaultdict(int)
    dataset_correct, dataset_total = defaultdict(int), defaultdict(int)

    for result in results:
        # Overall accuracy
        total_correct += result["overall"]
        total_examples += result["total"]

        # Category accuracy / Dataset accuracy
        for key, score in result.items():
            if key in Categories:
                category_correct[key] += score
                category_total[key] += result["total"]
            elif key != "overall":
                dataset_correct[key] += score
                dataset_total[key] += result["total"]

    overall_accuracy = (total_correct / total_examples) * 100 if total_examples > 0 else 0.0
    category_accuracy = {category: (category_correct[category] / category_total[category]) * 100 if category_total[category] > 0 else 0.0 for category in category_correct}
    dataset_accuracy = {dataset: (dataset_correct[dataset] / dataset_total[dataset]) * 100 if dataset_total[dataset] > 0 else 0.0 for dataset in dataset_correct}

    # eval_logger.info("=" * 50)
    # eval_logger.info(f"Overall Accuracy: {overall_accuracy:.2f}%")
    # eval_logger.info("Category-wise Accuracy:")
    # for category, acc in category_accuracy.items():
    #     eval_logger.info(f"  {category}: {acc:.2f}")
    # eval_logger.info("=" * 50)

    # # appending the results to the log file
    # with open('log_results.txt', 'a') as f:
    #     f.write("=" * 50 + "\n")
    #     f.write(f"Total Examples: {total_examples}\n")
    #     f.write(f"Overall Accuracy: {overall_accuracy:.2f}%\n")
    #     f.write("Category-wise Accuracy:\n")
    #     for category, acc in category_accuracy.items():
    #         f.write(f"  {category}: {acc:.2f}\n")
    #     f.write("=" * 50 + "\n")

    return round(overall_accuracy, 5)


def _aggregate_category_acc(results, target_category: str) -> float:
    total_correct = 0
    total_examples = 0
    for r in results:
        if r["category"] == target_category:
            total_correct += r["score"]
            total_examples += 1
    return round((total_correct / total_examples) * 100, 5) if total_examples > 0 else 0.0


def _aggregate_category_caa(results, target_category: str) -> float:
    total_adjusted = 0.0
    total_baseline = 0.0
    for r in results:
        if r["category"] == target_category:
            total_adjusted += r["score"]
            total_baseline += r["total"]
    return round((total_adjusted / total_baseline) * 100, 5) if total_baseline > 0 else 0.0


def aggregate_3d_information_understanding_acc(results):
    return _aggregate_category_acc(results, "3d information understanding")


def aggregate_3d_information_understanding_caa(results):
    return _aggregate_category_caa(results, "3d information understanding")


def aggregate_counting_and_existence_acc(results):
    return _aggregate_category_acc(results, "counting & existence")


def aggregate_counting_and_existence_caa(results):
    return _aggregate_category_caa(results, "counting & existence")


def aggregate_movement_prediction_and_navigation_acc(results):
    return _aggregate_category_acc(results, "movement prediction & navigation")


def aggregate_movement_prediction_and_navigation_caa(results):
    return _aggregate_category_caa(results, "movement prediction & navigation")


def aggregate_multiview_and_crossimage_reasoning_acc(results):
    return _aggregate_category_acc(results, "multi-view & cross-image reasoning")


def aggregate_multiview_and_crossimage_reasoning_caa(results):
    return _aggregate_category_caa(results, "multi-view & cross-image reasoning")


def aggregate_object_localization_and_positioning_acc(results):
    return _aggregate_category_acc(results, "object localization & positioning")


def aggregate_object_localization_and_positioning_caa(results):
    return _aggregate_category_caa(results, "object localization & positioning")


def aggregate_spatial_relationship_reasoning_acc(results):
    return _aggregate_category_acc(results, "spatial relationship reasoning")


def aggregate_spatial_relationship_reasoning_caa(results):
    return _aggregate_category_caa(results, "spatial relationship reasoning")
