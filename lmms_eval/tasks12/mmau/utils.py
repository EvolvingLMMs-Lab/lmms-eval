import datetime
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file


def doc_to_audio(doc):
    return [doc["audio"]]


def doc_to_text(doc, lmms_eval_specific_kwargs):
    letter = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    question = doc["question"]
    choices = json.loads(doc["choices"])
    choices = "\n".join([f"{letter[i]}. {choice}" for i, choice in enumerate(choices)])
    return f"{pre_prompt}{question}\n{choices}{post_prompt}"


def doc_to_choice(doc):
    choices = json.loads(doc["choices"])
    return choices


def mmau_process_results(doc, result):
    letter = ["A", "B", "C", "D"]
    response = parse_multi_choice_response(result[0], letter)
    response = letter_to_ans(response, json.loads(doc["choices"]))
    doc["model_prediction"] = response
    response = response.strip().lower()
    gt_ans = doc["answer"].strip().lower()
    score = 1.0 if response == gt_ans else 0.0

    return {"accuracy": {"overall": score, "task": doc["task"]}, "submission": {**doc}}


def mmau_aggregate_results(results):
    total_correct = 0
    group_totals = defaultdict(int)
    group_correct = defaultdict(int)

    for result in results:
        accuracy = result["overall"]
        total_correct += accuracy

        group_totals[result["task"]] += 1
        group_correct[result["task"]] += accuracy

    overall_accuracy = round(total_correct * 100 / len(results), 5)
    categorical_accuracy = {key: round(group_correct[key] * 100 / group_totals[key], 5) for key in group_totals.keys()}
    eval_logger.info("=" * 50)
    eval_logger.info(f"Overall accuracy: {overall_accuracy}")
    eval_logger.info("Categorical accuracy: ")
    for key, value in categorical_accuracy.items():
        eval_logger.info(f"{key} accuracy: {value}")
    eval_logger.info("=" * 50)
    return overall_accuracy


def mmau_aggregate_results_for_submission(results, args):
    path = generate_submission_file("mmau_submission.json", args)
    filtered_results = []
    keys_to_keep = ["id", "audio_id", "question", "choices", "model_prediction", "dataset", "task", "split", "category", "sub-category", "difficulty"]

    for result in results:
        filtered_result = {key: result[key] for key in keys_to_keep if key in result}
        filtered_results.append(filtered_result)

    results = filtered_results
    with open(path, "w") as f:
        json.dump(results, f, indent=4)
    eval_logger.info(f"Results saved to {path}.")


def parse_multi_choice_response(response, all_choices):
    """
    Parse the prediction from the generated response.
    Return the predicted choice letter e.g., A, B, C, D.
    """
    # Clean response of unwanted characters
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
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


def letter_to_ans(letter, choices):
    return choices[ord(letter) - ord("A")]
