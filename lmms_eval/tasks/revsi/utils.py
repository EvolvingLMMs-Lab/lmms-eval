import os
import datasets
import numpy as np
import pandas as pd
from huggingface_hub.constants import HF_HOME
from lmms_eval.utils import resolve_cache_dir
from lmms_eval.tasks._task_utils.default_template_yaml import load_default_template_yaml


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


REVSI_METRICS = [
    "overall_acc",
    "object_abs_distance_acc",
    "object_counting_acc",
    "object_rel_direction_acc",
    "object_rel_distance_acc",
    "object_size_estimation_acc",
    "room_size_estimation_acc",
    "route_planning_acc",
]


COMPOSITE_METRICS = {
    "object_rel_direction_acc": [
        "object_rel_direction_forward_easy",
        "object_rel_direction_backward_easy",
        "object_rel_direction_forward_hard",
        "object_rel_direction_backward_hard",
    ],
    "object_rel_distance_acc": [
        "object_rel_distance_closest",
        "object_rel_distance_farthest",
    ],
    "object_counting_acc": [
        "object_counting_single",
        "object_counting_multiple",
    ],
    "room_size_estimation_acc": [
        "room_size_estimation_single",
        "room_size_estimation_multiple",
    ],
}


config = load_default_template_yaml(__file__)
cache_dir = resolve_cache_dir(config["dataset_kwargs"]["cache_dir"], base_dir=HF_HOME)


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
        return dataset.shuffle(seed=42)
    return dataset


def _mean_relative_accuracy(pred, target, start, end, interval):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    acc = (abs(pred - target) / target) <= (1 - conf_intervs)
    return acc.mean()


def revsi_process_results(doc, results):
    pred_answer = str(results[0]).strip().split(" ")[0].rstrip(".").strip()
    gt_answer = doc["ground_truth"]
    if doc["question_type"] in MCQ_QUESTION_TYPES:
        acc = 1.0 if pred_answer.lower() == gt_answer.lower() else 0.0
    elif doc["question_type"] in NQ_QUESTION_TYPES:
        try:
            acc = _mean_relative_accuracy(float(pred_answer), float(gt_answer), 0.5, 0.95, 0.05)
        except:
            acc = 0.0
    doc["acc"] = acc
    return {metric: doc for metric in REVSI_METRICS}


def _collapse_question_types(output, metric_name, question_types):
    question_type_metrics = [
        f"{question_type}_acc" for question_type in question_types if f"{question_type}_acc" in output
    ]
    if not question_type_metrics:
        return
    output[metric_name] = np.mean([output.pop(metric) for metric in question_type_metrics])


def _compute_all_subscores(results) -> dict:
    results = pd.DataFrame(results)
    output = {
        f"{question_type}_acc": per_question_type["acc"].mean()
        for question_type, per_question_type in results.groupby("question_type")
    }

    for metric_name, question_types in COMPOSITE_METRICS.items():
        _collapse_question_types(output, metric_name, question_types)

    output["overall_acc"] = sum(output.values()) / len(output) if output else 0.0
    return output


def _aggregate_metric(results, metric_name):
    return _compute_all_subscores(results).get(metric_name, 0.0)


def revsi_aggregate_overall(results):
    return _aggregate_metric(results, "overall_acc")


def revsi_aggregate_object_abs_distance_acc(results):
    return _aggregate_metric(results, "object_abs_distance_acc")


def revsi_aggregate_object_counting_acc(results):
    return _aggregate_metric(results, "object_counting_acc")


def revsi_aggregate_object_rel_direction_acc(results):
    return _aggregate_metric(results, "object_rel_direction_acc")


def revsi_aggregate_object_rel_distance_acc(results):
    return _aggregate_metric(results, "object_rel_distance_acc")


def revsi_aggregate_object_size_estimation_acc(results):
    return _aggregate_metric(results, "object_size_estimation_acc")


def revsi_aggregate_room_size_estimation_acc(results):
    return _aggregate_metric(results, "room_size_estimation_acc")


def revsi_aggregate_route_planning_acc(results):
    return _aggregate_metric(results, "route_planning_acc")
