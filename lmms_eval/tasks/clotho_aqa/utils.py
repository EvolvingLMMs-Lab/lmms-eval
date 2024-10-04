import json
import os

from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file


def clotho_aqa_doc_to_audio(doc):
    return [doc["audio"]["array"]]

def clotho_aqa_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"

def parse_pred_ans(pred_ans):
    """Brought from Otter Eval"""
    pred_ans = pred_ans.lower().strip().replace(".", "")
    pred_label = None
    if len(pred_ans) == 1:
        if pred_ans == "y":
            pred_label = "yes"
        elif pred_ans == "n":
            pred_label = "no"
        else:
            pred_label = pred_ans
    else:
        if "yes" in pred_ans:
            pred_label = "yes"
        elif "no" in pred_ans:
            pred_label = "no"
        else:
            pred_label = pred_ans
    return pred_label

def clotho_aqa_process_results(doc, results):
    pred = results[0]
    pred_ans = parse_pred_ans(pred)
    gt_ans = doc["answer"].lower().strip().replace(".", "")
    score = 1.0 if pred_ans == gt_ans else 0.0
    return {"exact_match": {"score": score}}

def clotho_aqa_aggregate_results(results):
    correct = 0.0
    total = 0.0
    for result in results:
        correct += result["score"]
        total += 1

    return correct / total * 100.0
