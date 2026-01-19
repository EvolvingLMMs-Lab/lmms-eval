import datetime
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import requests
import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file


def muchomusic_doc_to_audio(doc):
    return [doc["context"]]


def muchomusic_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["instruction"]
    answers = doc["choices"]
    question = f"{question}\n{answers}"
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"


def muchomusic_doc_to_target(doc):
    return doc["answer"][1]


def muchomusic_doc_to_choice(doc):
    return ["A", "B", "C", "D"]


def muchomusic_process_results(doc, result):
    response = result[0].strip()
    all_choices = ["A", "B", "C", "D"]
    pred = parse_multi_choice_response(response, all_choices)  # AdaptfromMMMU
    gt_ans = doc["answer"][1]
    score = 1.0 if pred == gt_ans else 0.0
    return {"accuracy": score}


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
