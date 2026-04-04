import os
import yaml
import datasets
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger as eval_logger


MCQ_QUESTION_TYPES = [
    "object_rel_direction_forward_easy",
    "object_rel_direction_backward_easy",
    "object_rel_direction_forward_hard",
    "object_rel_direction_backward_hard",
    "object_rel_distance_closest",
    "object_rel_distance_farthest",
    "route_planning",
]


NQ_QUESTION_TYPES = [
    "object_counting_single",
    "object_counting_multiple",
    "object_abs_distance",
    "object_size_estimation",
    "room_size_estimation_single",
    "room_size_estimation_multiple"
]


hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
config = yaml.safe_load("".join(safe_data))
cache_name = config["dataset_kwargs"]["cache_dir"]
dataset_name = config["dataset_name"]
cache_dir = os.path.join(base_cache_dir, cache_name)


def revsi_doc_to_visual(doc):
    video_path = os.path.join(cache_dir, f"{doc['num_frames']}_frame", f"{doc['scene_id']}.mp4")
    if not os.path.exists(video_path):
        raise FileExistsError(f"video path:{video_path} does not exist.")
    return [video_path]


def revsi_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    if doc["question_type"] in NQ_QUESTION_TYPES:
        post_prompt = lmms_eval_specific_kwargs.get("nq_post_prompt", "")
        return "\n".join([pre_prompt, question, post_prompt]).strip()
    elif doc["question_type"] in MCQ_QUESTION_TYPES:
        options = "Options:\n" + "\n".join(doc["options"])
        post_prompt = lmms_eval_specific_kwargs.get("mcq_post_prompt", "")
        return "\n".join([pre_prompt, question, options, post_prompt]).strip()


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    if os.getenv("LMMS_EVAL_SHUFFLE_DOCS", None):
        eval_logger.info(f"Environment variable LMMS_EVAL_SHUFFLE_DOCS detected, dataset will be shuffled.")
        return dataset.shuffle(seed=42)
    return dataset


def _mean_relative_accuracy(pred, target, start, end, interval):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = (abs(pred - target) / target) <= (1 - conf_intervs)
    return accuracy.mean()


def revsi_process_results(doc, results):
    pred_answer = str(results[0]).strip().split(" ")[0].rstrip(".").strip()
    gt_answer = doc["ground_truth"]
    if doc["question_type"] in MCQ_QUESTION_TYPES:
        accuracy = 1.0 if pred_answer.lower() == gt_answer.lower() else 0.0
    elif doc["question_type"] in NQ_QUESTION_TYPES:
        try:
            accuracy = _mean_relative_accuracy(float(pred_answer), float(gt_answer), 0.5, 0.95, 0.05)
        except:
            accuracy = 0.0
    doc["accuracy"] = accuracy
    return {"revsi_acc": doc}


def revsi_aggregate_results(results):
    results = pd.DataFrame(results)
    output = {}
    for question_type, question_type_idx in results.groupby("question_type").groups.items():
        per_question_type = results.iloc[question_type_idx]
        output[f"{question_type}_accuracy"] = per_question_type["accuracy"].mean()

    rel_dir_accs = [
        output.pop("object_rel_direction_forward_easy_accuracy"),
        output.pop("object_rel_direction_backward_easy_accuracy"),
        output.pop("object_rel_direction_forward_hard_accuracy"),
        output.pop("object_rel_direction_backward_hard_accuracy"),
    ]
    output["object_rel_direction_accuracy"] = np.mean(rel_dir_accs)

    rel_dist_accs = [
        output.pop("object_rel_distance_closest_accuracy"),
        output.pop("object_rel_distance_farthest_accuracy"),
    ]
    output["object_rel_distance_accuracy"] = np.mean(rel_dist_accs)

    obj_count_accs = [
        output.pop("object_counting_single_accuracy"),
        output.pop("object_counting_multiple_accuracy"),
    ]
    output["object_counting_accuracy"] = np.mean(obj_count_accs)

    if "room_size_estimation_single_accuracy" in output and "room_size_estimation_multiple_accuracy" in output:
        room_size_accs = [
            output.pop("room_size_estimation_single_accuracy"),
            output.pop("room_size_estimation_multiple_accuracy"),
        ]
        output["room_size_estimation_accuracy"] = np.mean(room_size_accs)

    output["overall"] = sum(output.values()) / len(output)
    eval_logger.info(f"Evaluation results: {output}")
    return output
