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
import requests
import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file


def doc_to_audio(doc):
    return [doc["audio"]]


def doc_to_text(doc, lmms_eval_specific_kwargs):
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{post_prompt}"


classes = ["Laughter", "Sniff", "Throat", "Cough", "Sigh", "Sneeze"]


def doc_to_choice(doc):
    return ["Laughter", "Sniff", "Throat clearing", "Cough", "Sigh", "Sneeze"]


def vocalsound_process_results(doc, result):
    response = result[0].strip()
    gt_ans = doc["answer"]
    pred = get_answer(response)
    score = 1.0 if pred == gt_ans else 0.0
    return {"accuracy": {"overall": score, "age": doc["age_group"], "spk_id": doc["spk_id"]}}


def vocalsound_aggregate_results(results):
    total_correct = 0
    group_totals = defaultdict(int)
    group_correct = defaultdict(int)

    for result in results:
        accuracy = result["overall"]
        total_correct += accuracy

        # Gender grouping
        if result["spk_id"][0] == "f":
            group_totals["female"] += 1
            group_correct["female"] += accuracy
        else:
            group_totals["male"] += 1
            group_correct["male"] += accuracy

        # Age grouping
        age_group = f"age{str(result['age'])}"
        group_totals[age_group] += 1
        group_correct[age_group] += accuracy

    overall_accuracy = round(total_correct / len(results), 5)
    categorical_accuracy = {
        "male_accuracy": round(group_correct["male"] / group_totals.get("male", 1), 5),  # Avoid division by zero
        "female_accuracy": round(group_correct["female"] / group_totals.get("female", 1), 5),
        "age_18_25_accuracy": round(group_correct["age1"] / group_totals.get("age1", 1), 5),
        "age_26_48_accuracy": round(group_correct["age2"] / group_totals.get("age2", 1), 5),
        "age_49_80_accuracy": round(group_correct["age3"] / group_totals.get("age3", 1), 5),
    }
    eval_logger.info("=" * 50)
    eval_logger.info(f"Overall accuracy: {overall_accuracy}")
    eval_logger.info("Categorical accuracy: ")
    for key, value in categorical_accuracy.items():
        eval_logger.info(f"{key} accuracy: {value}")
    eval_logger.info("=" * 50)
    return overall_accuracy


def get_answer(response):
    for temp in classes:
        if temp.lower() in response.lower():
            return temp if temp != "Throat" else "Throat clearing"
