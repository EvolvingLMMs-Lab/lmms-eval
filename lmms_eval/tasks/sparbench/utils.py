import math
import os
import re
from functools import partial
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import yaml
from loguru import logger as eval_logger

MCA_QUESTION_TYPES = [
    "obj_spatial_relation_oo",
    "obj_spatial_relation_oc_mv",
    "obj_spatial_relation_oo_mv",
    "spatial_imagination_oc",
    "spatial_imagination_oo",
    "spatial_imagination_oc_mv",
    "spatial_imagination_oo_mv",
    "position_matching",
    "camera_motion_infer",
    "distance_infer_center_oo",
    "distance_infer_center_oo_mv",
]
NA_QUESTION_TYPES = [
    "depth_prediction_oc",
    "depth_prediction_oo",
    "distance_prediction_oc",
    "distance_prediction_oo",
    "depth_prediction_oc_mv",
    "depth_prediction_oo_mv",
    "distance_prediction_oo_mv",
    "distance_prediction_oc_mv",
]


SPECIAL_QUESTION_TYPES = [
    "view_change_infer",
]

METRICS_FOR_NA = {
    "MRA:.5:.95:.05": "partial(mean_relative_accuracy, start=.5, end=.95, interval=.05)",
}

METRICS_FOR_MCA = {
    "accuracy": "exact_match",
}

Low = [
    "depth_prediction_oc",
    "depth_prediction_oo",
    "distance_prediction_oc",
    "distance_prediction_oo",
    "depth_prediction_oc_mv",
    "depth_prediction_oo_mv",
    "distance_prediction_oo_mv",
    "distance_prediction_oc_mv",
]

Middle = [
    "view_change_infer",
    "position_matching",
    "camera_motion_infer",
]

High = [
    "obj_spatial_relation_oo",
    "obj_spatial_relation_oc_mv",
    "obj_spatial_relation_oo_mv",
    "spatial_imagination_oc",
    "spatial_imagination_oo",
    "spatial_imagination_oc_mv",
    "spatial_imagination_oo_mv",
    "distance_infer_center_oo",
    "distance_infer_center_oo_mv",
]

with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


def sparbench_doc_to_visual(doc):
    image = doc["image"]
    return image


def sparbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")  # or "These are frames of a video."

    if doc["task"] in NA_QUESTION_TYPES:
        post_prompt = lmms_eval_specific_kwargs.get("na_post_prompt", "") or "Please answer the question using a single word or phrase."
        return pre_prompt + "\n" + question + "\n" + post_prompt
    elif doc["task"] in MCA_QUESTION_TYPES:
        post_prompt = ""
        if doc["task"] in ["position_matching", "camera_motion_infer"]:
            post_prompt = "The values represent the bounding box coordinates normalized to a 0-1000 scale, with the top-left corner as the origin of the image."
        post_prompt2 = "Answer with the option's letter from the given choices directly."
        return pre_prompt + "\n" + question + "\n" + post_prompt + "\n" + post_prompt2
    elif doc["task"] in SPECIAL_QUESTION_TYPES:
        post_prompt1 = ""
        post_prompt2 = ""
        return pre_prompt + "\n" + question + "\n" + post_prompt1 + "\n" + post_prompt2
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    if os.getenv("LMMS_EVAL_SHUFFLE_DOCS", None):
        eval_logger.info(f"Environment variable LMMS_EVAL_SHUFFLE_DOCS detected, dataset will be shuffled.")
        return dataset.shuffle(seed=42)
    return dataset


def fuzzy_matching(pred):
    return pred.split(" ")[0].rstrip(".").strip()


def process_na(pred, task):
    numbers = re.findall(r"(?<!\^)\d+\.\d+|(?<!\^)\d+", pred)

    # Convert the matched numbers to float or int
    extracted_numbers = [float(num) if "." in num else int(num) for num in numbers]
    if task in [
        "depth_prediction_oc_mv",
        "depth_prediction_oo_mv",
        "distance_prediction_oc_mv",
        "distance_prediction_oo_mv",
    ]:
        if len(extracted_numbers) == 0:
            extracted_numbers = [-1]
        extracted_numbers = [extracted_numbers[-1]]
    return extracted_numbers[0]


def calculate_distance(coord1, coord2):
    return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)


def parse_instruction(instruction):
    return {k: float(v) for k, v in [item.split(":") for item in instruction.split(",")]}


def compute_vci_metric(pred, answer):

    acion_list = ["move_right", "move_left", "move_forward", "move_backward", "move_up", "move_down", "rotate_right", "rotate_left", "rotate_up", "rotate_down"]
    action_order = ["move_right_left", "move_up_down", "move_forward_backward", "rotate_right_left", "rotate_up_down"]

    answer_dict = parse_instruction(pred)
    gt_dict = parse_instruction(answer)

    answer_list = []
    gt_list = []

    for action_pair in action_order:
        if action_pair == "move_right_left":
            answer_list.append(answer_dict.get("move_right", 0) - answer_dict.get("move_left", 0))
            gt_list.append(gt_dict.get("move_right", 0) - gt_dict.get("move_left", 0))
        elif action_pair == "move_up_down":
            answer_list.append(answer_dict.get("move_up", 0) - answer_dict.get("move_down", 0))
            gt_list.append(gt_dict.get("move_up", 0) - gt_dict.get("move_down", 0))
        elif action_pair == "move_forward_backward":
            answer_list.append(answer_dict.get("move_forward", 0) - answer_dict.get("move_backward", 0))
            gt_list.append(gt_dict.get("move_forward", 0) - gt_dict.get("move_backward", 0))
        elif action_pair == "rotate_right_left":
            answer_list.append(answer_dict.get("rotate_right", 0) - answer_dict.get("rotate_left", 0))
            gt_list.append(gt_dict.get("rotate_right", 0) - gt_dict.get("rotate_left", 0))
        elif action_pair == "rotate_up_down":
            answer_list.append(answer_dict.get("rotate_up", 0) - answer_dict.get("rotate_down", 0))
            gt_list.append(gt_dict.get("rotate_up", 0) - gt_dict.get("rotate_down", 0))

    mra_list = []
    for gt, answer in zip(gt_list, answer_list):
        mra = mean_relative_accuracy(gt, answer, start=0.5, end=0.95, interval=0.05)
        mra_list.append(mra)

    return np.mean(mra_list)


def parse_cmi(text):
    pattern = r"\([0-9\.]+,[0-9\.]+\)|[0-9\.]+"

    matches = re.findall(pattern, text)
    eval_logger.debug(f"[parse_cmi] input text: '{text}'")
    eval_logger.debug(f"[parse_cmi] initial matches: {matches}")

    if len(matches) < 2:
        eval_logger.warning(f"[parse_cmi] Less than 2 matches found, applying fallback logic")
        if len(matches) == 1 and "(" in matches[0]:
            matches.append("0.0")
            eval_logger.debug(f"[parse_cmi] Appended '0.0', matches now: {matches}")
        elif len(matches) == 1 and "." in matches[0]:
            matches.insert(0, "(0.0,0.0)")
            eval_logger.debug(f"[parse_cmi] Inserted '(0.0,0.0)', matches now: {matches}")

    result = []
    for match in matches:
        if "(" in match and ")" in match:
            num1, num2 = match.strip("()").split(",")
            result.extend([float(num1), float(num2)])
        else:
            result.append(float(match))

    eval_logger.debug(f"[parse_cmi] final result (len={len(result)}): {result}")
    if len(result) < 3:
        eval_logger.warning(f"[parse_cmi] Result has fewer than 3 elements! Accessing result[2] will fail.")
    return result


def compute_cmi_metric(pred, answer):
    eval_logger.debug(f"[compute_cmi_metric] pred: '{pred}', answer: '{answer}'")
    pred_process = parse_cmi(pred)
    ans_process = parse_cmi(answer)
    eval_logger.debug(f"[compute_cmi_metric] pred_process: {pred_process}, ans_process: {ans_process}")
    try:
        dist = math.sqrt((pred_process[0] / 1000 - ans_process[0] / 1000) ** 2 + (pred_process[1] / 1000 - ans_process[1] / 1000) ** 2 + (pred_process[2] - ans_process[2]) ** 2)
        eval_logger.debug(f"[compute_cmi_metric] computed distance: {dist} (note: lower is better, but raw distance returned)")
    except IndexError as e:
        eval_logger.error(f"[compute_cmi_metric] IndexError: {e}. pred_process has {len(pred_process)} elements, ans_process has {len(ans_process)} elements")
        raise
    return dist


def exact_match(pred, target):
    # return 1. if pred.lower() == target.lower() else 0.
    pred = pred.lower()
    target = target.lower()
    eval_logger.debug(f"[exact_match] pred: '{pred}', target: '{target}'")
    if pred.lower() == target.lower():
        eval_logger.debug(f"[exact_match] Matched via exact match")
        return 1.0
    elif pred in target:
        eval_logger.debug(f"[exact_match] Matched via 'pred in target'")
        return 1.0
    elif pred[0] == target:
        eval_logger.warning(f"[exact_match] SUSPICIOUS: Matched via pred[0]==target. pred[0]='{pred[0]}', target='{target}'. This compares first char to entire string!")
        return 1.0
    else:
        eval_logger.debug(f"[exact_match] No match")
        return 0


def abs_dist_norm(pred, target):
    if target == 0.0:
        return abs(pred - target)
    else:
        return abs((pred - target) / target)


def mean_relative_accuracy(pred, target, start, end, interval):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()


WORST_CASE_FOR_METRICS = {
    "accuracy": 0.0,
    "MRA:.5:.95:.05": 0.0,
}


def to_float(pred):
    try:
        pred = float(pred)
    except BaseException as e:
        pred = None
    return pred


def sparbench_process_results(doc, results):

    doc["prediction"] = results[0]
    if doc["task"] in MCA_QUESTION_TYPES:
        for key, value in METRICS_FOR_MCA.items():
            doc[key] = eval(value)(doc["prediction"], doc["answer"])
        pass
    elif doc["task"] in NA_QUESTION_TYPES:
        for key, value in METRICS_FOR_NA.items():
            try:
                doc[key] = eval(value)(to_float(process_na(doc["prediction"], doc["task"])), to_float(doc["answer"]))
            except:
                doc[key] = WORST_CASE_FOR_METRICS[key]
    elif doc["task"] in SPECIAL_QUESTION_TYPES:
        if doc["task"] == "view_change_infer":
            try:
                doc["vci_metric"] = compute_vci_metric(doc["prediction"], doc["answer"])
            except:
                doc["vci_metric"] = 0

    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")

    return {"sparbench_score": doc}


def sparbench_aggregate_results(results):
    results = pd.DataFrame(results)
    output = {}
    for question_type, question_type_indexes in results.groupby("task").groups.items():
        per_question_type = results.iloc[question_type_indexes]

        if question_type in MCA_QUESTION_TYPES:
            for metric in METRICS_FOR_MCA.keys():
                output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
        elif question_type in NA_QUESTION_TYPES:
            for metric in METRICS_FOR_NA.keys():
                if metric == "success_rate":
                    output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
                else:
                    output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
        elif question_type in SPECIAL_QUESTION_TYPES:
            if question_type == "view_change_infer":
                output[f"{question_type}_vci_metric"] = per_question_type["vci_metric"].mean()

    output["overall"] = sum([_ for _ in output.values()]) / len(output)
    # eval_logger.info(f"Evaluation results: {output}")
    low_list = []
    middle_list = []
    high_list = []
    for task in output:
        task_name = "_".join(task.split("_")[:-1])
        if task_name in Low:
            low_list.append(output[task])
        elif task_name in Middle:
            middle_list.append(output[task])
        elif task_name in High:
            high_list.append(output[task])

    output["Low"] = np.mean(low_list)
    output["Middle"] = np.mean(middle_list)
    output["High"] = np.mean(high_list)

    eval_logger.info(f"Evaluation results: {output}")
    return output["overall"] * 100.0
