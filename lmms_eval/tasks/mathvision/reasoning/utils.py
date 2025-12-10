import json
import os
import time
from pathlib import Path

import pandas as pd
import requests
import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.reasoning_utils import compute_score

SYSTEM_PROMPT = (
    "You are a helpful assistant. When the user asks a question, your response must include two parts: "
    "first, the reasoning process enclosed in <think>...</think> tags, then the final answer enclosed in <answer>...</answer> tags."
    "Please provide a clear, concise response within <answer> </answer> tags that directly addresses the question."
)


def mathvision_doc_to_visual(doc):
    return [doc["decoded_image"].convert("RGB")]


def mathvision_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question, choices = doc["question"], doc["options"]
    len_choices = len(choices)
    options = [chr(ord("A") + i) for i in range(len_choices)]
    choices_str = "\n".join([f"{option}. {choice}" for option, choice in zip(options, choices)])

    mc_prompt = ""
    if lmms_eval_specific_kwargs is not None:
        mc_prompt = "\n" + lmms_eval_specific_kwargs["mc_prompt"]

    query_prompt = 'Please solve the problem step by step and put your answer in one "\\boxed{}".'
    if choices_str:
        query_prompt += f"{question}\nChoices: {choices_str}" + mc_prompt
    else:
        query_prompt += question
    return query_prompt


def mathvision_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    query_prompt = mathvision_doc_to_text(doc, lmms_eval_specific_kwargs)
    system_messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
    visuals = mathvision_doc_to_visual(doc)

    user_messages = []
    user_messages.append({"role": "user", "content": [{"type": "image", "url": visuals[0]}, {"type": "text", "text": query_prompt}]})

    return system_messages + user_messages


def mathvision_process_results(doc, results):
    acc_score = 0
    format_score = 0
    question = mathvision_doc_to_text(doc, None)
    extra_info = {"question": question}
    for pred in results:
        score_dict = compute_score(data_source="mathvista", solution_str=pred.strip(), ground_truth=doc["answer"], extra_info=extra_info)
        acc_score += score_dict["acc_score"]
        format_score += score_dict.get("format_reward_score", 0.0)

    return {"acc_score": acc_score / len(results) if results else 0.0, "format_score": format_score / len(results) if results else 0.0}
