import json
import random
import re
from collections import defaultdict

import numpy as np
from loguru import logger as eval_logger

MULTI_CHOICE_PROMPT = "请只回答选项字母（如：A）。"
DIRECT_PROMPT = "请直接回答证候名称，不需要解释。"

def parse_options(options):
    """Parse options into formatted string with A, B, C, D, E format"""
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    choices_str = "\n".join([f"{option_letter}. {option}" for option_letter, option in zip(option_letters, options)])
    return choices_str


def tcm_sd_doc_to_text_multiple_choice(doc, lmms_eval_specific_kwargs=None):
    """Convert document to text prompt for TCM multiple choice questions"""
    prompt = doc["prompt"]
    full_prompt = f"{prompt}"
    return full_prompt


def tcm_sd_doc_to_text_direct(doc, lmms_eval_specific_kwargs=None):
    """Convert document to text prompt for TCM direct diagnosis"""
    prompt = doc["prompt"]
    full_prompt = f"{prompt}"
    return full_prompt


def tcm_sd_doc_to_text_rc_five(doc, lmms_eval_specific_kwargs=None):
    """Convert document to text prompt for TCM reading comprehension (five options)"""
    prompt = doc["prompt"]
    full_prompt = f"{prompt}"
    return full_prompt


def tcm_sd_doc_to_text_rc_all(doc, lmms_eval_specific_kwargs=None):
    """Convert document to text prompt for TCM reading comprehension (all options)"""
    prompt = doc["prompt"]
    full_prompt = f"{prompt}"
    return full_prompt


def get_multi_choice_info(options):
    """
    Given the list of options for multiple choice question
    Return the index2ans and all_choices
    """
    start_chr = "A"
    all_choices = []
    index2ans = {}
    for i, option in enumerate(options):
        index2ans[chr(ord(start_chr) + i)] = option
        all_choices.append(chr(ord(start_chr) + i))
    return index2ans, all_choices


def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D, E.
    """
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "

    index_ans = True
    ans_with_brack = False
    candidates = []
    
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice}." in response:
                candidates.append(choice)

    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False

    if len(candidates) == 0:
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    start_indexes.append(response.rfind(f"({can})"))
            else:
                for can in candidates:
                    start_indexes.append(response.rfind(f" {can} "))
        else:
            for can in candidates:
                start_indexes.append(response.lower().rfind(index2ans[can].lower()))
        pred_index = candidates[np.argmax(start_indexes)]
    else:
        pred_index = candidates[0]
    return pred_index


def tcm_sd_process_results_multiple_choice(doc, results):
    """Process results for TCM multiple choice questions"""
    pred = results[0]
    options = doc["options"]
    index2ans, all_choices = get_multi_choice_info(options)
    parsed_pred = parse_multi_choice_response(pred, all_choices, index2ans)
    
    return {
        "tcm_sd_acc": {
            "user_id": doc["user_id"],
            "expected_answer": doc["expected_answer"],
            "parsed_pred": parsed_pred,
            "expected_syndrome": doc.get("expected_syndrome", "")
        },
        "submission": {doc["user_id"]: pred}
    }


def tcm_sd_process_results_direct(doc, results):
    """Process results for TCM direct diagnosis"""
    pred = results[0].strip()
    return {
        "tcm_sd_acc": {
            "user_id": doc["user_id"],
            "expected_answer": doc["expected_answer"],
            "parsed_pred": pred
        },
        "submission": {doc["user_id"]: pred}
    }


def tcm_sd_process_results_rc_five(doc, results):
    """Process results for TCM reading comprehension (five options)"""
    pred = results[0].strip()
    return {
        "tcm_sd_acc": {
            "user_id": doc["user_id"],
            "expected_answer": doc["expected_answer"],
            "parsed_pred": pred
        },
        "submission": {doc["user_id"]: pred}
    }


def tcm_sd_process_results_rc_all(doc, results):
    """Process results for TCM reading comprehension (all options)"""
    pred = results[0].strip()
    return {
        "tcm_sd_acc": {
            "user_id": doc["user_id"],
            "expected_answer": doc["expected_answer"],
            "parsed_pred": pred
        },
        "submission": {doc["user_id"]: pred}
    }


def eval_multi_choice(gold_i, pred_i):
    """Evaluate a multiple choice instance."""
    return gold_i == pred_i


def evaluate_tcm_sd_multiple_choice(samples):
    """Batch evaluation for TCM multiple choice questions."""
    pred_correct = 0
    for sample in samples:
        if eval_multi_choice(sample["expected_answer"], sample["parsed_pred"]):
            pred_correct += 1
    return {"acc": pred_correct / len(samples) if samples else 0}


def evaluate_tcm_sd_direct(samples):
    """Batch evaluation for TCM direct diagnosis"""
    pred_correct = 0
    for sample in samples:
        if sample["expected_answer"] == sample["parsed_pred"]:
            pred_correct += 1
    return {"acc": pred_correct / len(samples) if samples else 0}


def evaluate_tcm_sd_rc_five(samples):
    """Batch evaluation for TCM reading comprehension (five options)"""
    pred_correct = 0
    for sample in samples:
        if sample["expected_answer"] == sample["parsed_pred"]:
            pred_correct += 1
    return {"acc": pred_correct / len(samples) if samples else 0}


def evaluate_tcm_sd_rc_all(samples):
    """Batch evaluation for TCM reading comprehension (all options)"""
    pred_correct = 0
    for sample in samples:
        if sample["expected_answer"] == sample["parsed_pred"]:
            pred_correct += 1
    return {"acc": pred_correct / len(samples) if samples else 0}


def tcm_sd_aggregate_results_multiple_choice(results):
    """Aggregate results for TCM multiple choice questions"""
    metric_dict = evaluate_tcm_sd_multiple_choice(results)
    total_samples = len(results)
    accuracy = metric_dict["acc"]
    
    eval_logger.info(f"TCM SD Multiple Choice Evaluation Results:")
    eval_logger.info(f"Total samples: {total_samples}")
    eval_logger.info(f"Accuracy: {accuracy:.4f}")
    
    return accuracy


def tcm_sd_aggregate_results_direct(results):
    """Aggregate results for TCM direct diagnosis"""
    metric_dict = evaluate_tcm_sd_direct(results)
    total_samples = len(results)
    accuracy = metric_dict["acc"]

    eval_logger.info(f"TCM SD Direct Diagnosis Evaluation Results:")
    eval_logger.info(f"Total samples: {total_samples}")
    eval_logger.info(f"Accuracy: {accuracy:.4f}")

    return accuracy 


def tcm_sd_aggregate_results_rc_five(results):
    """Aggregate results for TCM reading comprehension (five options)"""
    metric_dict = evaluate_tcm_sd_rc_five(results)
    total_samples = len(results)
    accuracy = metric_dict["acc"]

    eval_logger.info(f"TCM SD Reading Comprehension (Five Options) Evaluation Results:")
    eval_logger.info(f"Total samples: {total_samples}")
    eval_logger.info(f"Accuracy: {accuracy:.4f}")

    return accuracy 


def tcm_sd_aggregate_results_rc_all(results):
    """Aggregate results for TCM reading comprehension (all options)"""
    metric_dict = evaluate_tcm_sd_rc_all(results)
    total_samples = len(results)
    accuracy = metric_dict["acc"]

    eval_logger.info(f"TCM SD Reading Comprehension (All Options) Evaluation Results:")
    eval_logger.info(f"Total samples: {total_samples}")
    eval_logger.info(f"Accuracy: {accuracy:.4f}")

    return accuracy 
