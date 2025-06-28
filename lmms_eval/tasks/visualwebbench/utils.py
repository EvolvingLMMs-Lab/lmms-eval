import os
import re
import sys

import numpy as np
import torch
from rouge import Rouge
from torchvision.ops import box_iou

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from loguru import logger as eval_logger
from prompts import DEFAULT_PROMPTS


def visualwebbench_doc_to_text(doc, lmms_eval_specific_kwargs):
    prompt = DEFAULT_PROMPTS[doc["task_type"]]

    if doc["task_type"] in ["web_caption", "heading_ocr"]:
        cur_prompt = prompt
    elif doc["task_type"] == "webqa":
        cur_prompt = prompt.format(question=doc["question"])
    elif doc["task_type"] == "element_ocr":
        cur_prompt = prompt.format(bbox_ratio=doc["bbox"])
    elif doc["task_type"] == "element_ground":
        cur_prompt = prompt.format(element_desc=doc["elem_desc"])
    elif doc["task_type"] == "action_prediction":
        cur_prompt = prompt.format(bbox_ratio=doc["bbox"], choices_text=doc["options"])
    elif doc["task_type"] == "action_ground":
        cur_prompt = prompt.format(instruction=doc["instruction"])

    return cur_prompt


def visualwebbench_doc_to_choice(doc):
    if "choices" in doc:
        return doc["choices"]
    else:
        return []


def visualwebbench_doc_to_target(doc):
    if "answer" in doc:
        return doc["answer"]
    elif "target" in doc:
        return doc["target"]
    else:
        return ""


def visualwebbench_doc_to_visual(doc):
    return [doc["image"]]


def visualwebbench_process_results_rouge(doc, results):
    pred = results[0] if len(results) > 0 else ""
    gold = doc["answer"] if "answer" in doc else doc.get("target", "")

    task_type = doc.get("task_type", "")

    if task_type == "web_caption":
        rouge_scores = eval_web_caption([pred], [gold])
    elif task_type == "heading_ocr":
        rouge_scores = eval_heading_ocr([pred], [gold])
    elif task_type == "element_ocr":
        rouge_scores = eval_element_ocr([pred], [gold])
    elif task_type == "web_caption":
        rouge_scores = eval_web_caption([pred], [gold])
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    return {"rouge_1": rouge_scores["rouge_1"], "rouge_2": rouge_scores["rouge_2"], "rouge_l": rouge_scores["rouge_l"]}


def visualwebbench_process_results_accuracy(doc, results):
    """Process results for accuracy-based tasks (action_ground, action_prediction, element_ground)"""
    pred = results[0] if len(results) > 0 else ""
    gold = doc["answer"] if "answer" in doc else doc.get("target", 0)

    # Parse multiple choice response and compute accuracy
    cur_pred = parse_multi_choice_response(pred, [chr(ord("A") + i) for i in range(8)])
    try:
        if ord("A") <= ord(cur_pred) <= ord("Z"):
            cur_pred = ord(cur_pred) - ord("A")
        else:
            cur_pred = -1
    except:
        cur_pred = -1

    is_correct = cur_pred == gold

    return {"accuracy": 100.0 if is_correct else 0.0}


def visualwebbench_process_results_f1(doc, results):
    """Process results for F1-based tasks (webqa)"""
    pred = results[0] if len(results) > 0 else ""
    gold = doc["answer"] if "answer" in doc else doc.get("target", [])

    # Make sure gold is a list
    if not isinstance(gold, list):
        gold = [gold] if gold else [""]

    # Compute F1 score directly
    f1_result = eval_webqa([pred], [gold])

    return {"f1_score": f1_result["f1"]}


def visualwebbench_rouge_1_aggregate(results):
    return sum(results) / len(results)


def visualwebbench_rouge_2_aggregate(results):
    return sum(results) / len(results)


def visualwebbench_rouge_l_aggregate(results):
    return sum(results) / len(results)


def visualwebbench_accuracy_aggregate(results):
    return sum(results) / len(results)


def visualwebbench_f1_score_aggregate(results):
    return sum(results) / len(results)


# Evaluation functions (from user's provided code)
def eval_web_caption(preds, golds, **kwargs):
    assert len(preds) == len(golds)
    for i in range(len(preds)):
        if not preds[i]:
            preds[i] = " "

    rouge = Rouge(metrics=["rouge-1", "rouge-2", "rouge-l"])
    scores = rouge.get_scores(preds, golds, avg=True)
    return dict(rouge_1=scores["rouge-1"]["f"] * 100, rouge_2=scores["rouge-2"]["f"] * 100, rouge_l=scores["rouge-l"]["f"] * 100)


def eval_heading_ocr(preds, golds, **kwargs):
    assert len(preds) == len(golds)
    for i in range(len(preds)):
        if not preds[i]:
            preds[i] = " "

    rouge = Rouge(metrics=["rouge-1", "rouge-2", "rouge-l"])
    scores = rouge.get_scores(preds, golds, avg=True)
    return dict(rouge_1=scores["rouge-1"]["f"] * 100, rouge_2=scores["rouge-2"]["f"] * 100, rouge_l=scores["rouge-l"]["f"] * 100)


def eval_element_ocr(preds, golds, **kwargs):
    assert len(preds) == len(golds)
    for i in range(len(preds)):
        if not preds[i] or len(preds[i]) == 1:
            preds[i] = " "

    rouge = Rouge()
    scores = rouge.get_scores(preds, golds, avg=True)
    return dict(rouge_1=scores["rouge-1"]["f"] * 100, rouge_2=scores["rouge-2"]["f"] * 100, rouge_l=scores["rouge-l"]["f"] * 100)


def eval_action_prediction(preds, golds, **kwargs):
    results = []
    for pred, gold in zip(preds, golds):
        cur_pred = parse_multi_choice_response(pred, [chr(ord("A") + i) for i in range(8)])
        try:
            if ord("A") <= ord(cur_pred) <= ord("Z"):
                cur_pred = ord(cur_pred) - ord("A")
            else:
                cur_pred = -1
        except:
            cur_pred = -1
        results.append(cur_pred == gold)

    return dict(accuracy=sum(results) / len(results) * 100)


def eval_element_ground(preds, golds, **kwargs):
    results = []
    for pred, gold in zip(preds, golds):
        cur_pred = parse_multi_choice_response(pred, [chr(ord("A") + i) for i in range(8)])
        try:
            if ord("A") <= ord(cur_pred) <= ord("Z"):
                cur_pred = ord(cur_pred) - ord("A")
            else:
                cur_pred = -1
        except:
            cur_pred = -1
        results.append(cur_pred == gold)

    return dict(accuracy=sum(results) / len(results) * 100)


def eval_action_ground(preds, golds, **kwargs):
    results = []
    for pred, gold in zip(preds, golds):
        cur_pred = parse_multi_choice_response(pred, [chr(ord("A") + i) for i in range(8)])
        try:
            if ord("A") <= ord(cur_pred) <= ord("Z"):
                cur_pred = ord(cur_pred) - ord("A")
            else:
                cur_pred = -1
        except:
            cur_pred = -1
        results.append(cur_pred == gold)

    return dict(accuracy=sum(results) / len(results) * 100)


def eval_webqa(preds, golds, **kwargs):
    f1_scores = []
    rouge = Rouge(metrics=["rouge-1"])
    for pred, gold_list in zip(preds, golds):
        try:
            if not pred:
                pred = " "
            cur_f1 = max([rouge.get_scores([pred], [gold], avg=True)["rouge-1"]["f"] for gold in gold_list])
            f1_scores.append(cur_f1)
        except:
            pass

    return dict(f1=sum(f1_scores) / len(f1_scores) * 100)


def parse_multi_choice_response(response: str, all_choices):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    """
    if len(response) == 1:
        return response.upper()
    elif not response:
        return "a"
    elif re.match(r"[A-Z]\.", response):
        return response[0]

    for char in [",", ".", "!", "?", ";", ":", "'", '"']:
        response = response.replace(char, "")
    response = " " + response + " "  # add space to avoid partial match

    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f" {choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0:  # still not get answer
        pred_index = "z"
    elif len(candidates) > 1:
        start_indexes = []
        if ans_with_brack:
            for can in candidates:
                index = response.rfind(f"({can})")
                start_indexes.append(index)  # -1 will be ignored anyway
        else:
            for can in candidates:
                index = response.rfind(f" {can} ")
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index
