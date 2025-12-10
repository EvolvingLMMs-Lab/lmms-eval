import json
import os
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.reasoning_utils import compute_score
from lmms_eval.tasks.mathverse.mathverse_evals import MathVerseEvaluator

SYSTEM_PROMPT = (
    "You are a helpful assistant. When the user asks a question, your response must include two parts: "
    "first, the reasoning process enclosed in <think>...</think> tags, then the final answer enclosed in <answer>...</answer> tags."
    "Please provide a clear, concise response within <answer> </answer> tags that directly addresses the question."
)

with open(Path(__file__).parent / "mathverse_testmini.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

mathverse_evaluator = MathVerseEvaluator()


def mathverse_doc_to_visual(doc):
    if str(doc["image"]).strip() == "":
        return []
    return [doc["image"].convert("RGB")]


def mathverse_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    problem = {
        "question": doc["question"],
        "answer": doc["answer"] if "answer" in doc else None,
        "query_wo": doc["query_wo"],
        "query_cot": doc["query_cot"],
        "question_type": doc["question_type"],
        "problem_version": doc["problem_version"],
    }
    query_prompt = mathverse_evaluator.create_one_query(
        problem, examples=None, shot_num=0, shot_type=lmms_eval_specific_kwargs["shot_type"], hint=lmms_eval_specific_kwargs.get("hint", None), query_type=lmms_eval_specific_kwargs["query_type"]
    )
    return query_prompt


def mathverse_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    query_prompt = mathverse_doc_to_text(doc, lmms_eval_specific_kwargs)
    system_messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
    visuals = mathverse_doc_to_visual(doc)

    user_messages = []
    if visuals:
        user_messages.append({"role": "user", "content": [{"type": "image", "url": visuals[0]}, {"type": "text", "text": query_prompt}]})
    else:
        user_messages.append({"role": "user", "content": [{"type": "text", "text": query_prompt}]})

    return system_messages + user_messages


def mathverse_process_results(doc, results):
    acc_score = 0
    format_score = 0
    # Build question text for extra_info using task's doc_to_text
    # mathverse_doc_to_text requires lmms_eval_specific_kwargs; default to config defaults if available
    try:
        default_kwargs = config.get("lmms_eval_specific_kwargs", {}).get("default", {})
    except Exception:
        default_kwargs = {}
    question = mathverse_doc_to_text(doc, default_kwargs)
    extra_info = {"question": question}
    for pred in results:
        score_dict = compute_score(data_source="mathvista", solution_str=pred.strip(), ground_truth=doc["answer"], extra_info=extra_info)
        acc_score += score_dict["acc_score"]
        format_score += score_dict.get("format_reward_score", 0.0)

    return {"acc_score": acc_score / len(results) if results else 0.0, "format_score": format_score / len(results) if results else 0.0}
