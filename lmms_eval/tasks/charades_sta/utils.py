import ast
import datetime
import json
import os
import re
import sys
from pathlib import Path

import yaml
from loguru import logger as eval_logger

import lmms_eval.tasks._task_utils.file_utils as file_utils

# with open(Path(__file__).parent / "_default_template.yaml", "r") as f:
#     raw_data = f.readlines()
#     safe_data = []
#     for i, line in enumerate(raw_data):
#         # remove function definition since yaml load cannot handle it
#         if "!function" not in line:
#             safe_data.append(line)

#     config = yaml.safe_load("".join(safe_data))


hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
# cache_dir = os.path.join(hf_home, cache_dir)
# base_cache_dir = config["dataset_kwargs"]["cache_dir"]
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "charades.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


# DATA_LIST = {
#     "charades": 'your_data_dir/Charades/',
# }
# Pass in video path here
# Can only work correctly with video llm
def temporal_grounding_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    video_path = doc["video"]
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = os.path.join(cache_dir, "Charades_v1_480", video_path)
    if os.path.exists(video_path):
        video_path = video_path
    elif "s3://" not in video_path:
        sys.exit(f"video path:{video_path} does not exist, please check")

    return [video_path]


# This is the place where you format your question
def temporal_grounding_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    question = doc["caption"]

    return f"{pre_prompt}{question}. {post_prompt}"


def temporal_grounding_doc_to_answer(doc):
    return doc["timestamp"]


# Process result for mcq answer generation
def temporal_grounding_process_results_generation(doc, result):
    pred = result[0]
    data_dict = {f'{doc["video"]}>>>{doc["caption"]}>>>{doc["timestamp"]}': pred}
    return {f"charades_sta_{metric}": data_dict for metric in CHARADES_STA_METRICS}


CHARADES_STA_METRICS = ["IOU@3", "IOU@5", "IOU@7", "mIOU"]


def extract_time(paragraph):
    prompt = "A specific example is : 20.8 - 30.0 seconds".lower()
    paragraph = paragraph.lower().replace(prompt, "").replace("to", "-")
    sentences = re.split(r"[!?\n]", paragraph)

    keywords = ["starts", "ends", "happens in", "start time", "end time", "start", "end", "happen"]
    candidates = [sentence for sentence in sentences if any(keyword in sentence for keyword in keywords)]
    if not candidates:
        candidates = sentences

    timestamps = []
    time_format_range_pattern = re.compile(r"\b(\d{1,2}:\d{2}(?::\d{2})?)\s*[–-]\s*(\d{1,2}:\d{2}(?::\d{2})?)\b")
    main_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*[–-]\s*(\d+(?:\.\d+)?)")
    time_number_pattern = re.compile(r"\b(\d+(?:\.\d+)?)\b")
    time_format_pattern = re.compile(r"\b(\d{1,2}:\d{2}(?::\d{2})?)\b")
    fallback_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*s?\s*[–-]\s*(\d+(?:\.\d+)?)\s*s?")

    for sentence in candidates:
        time_matches = time_format_range_pattern.findall(sentence)
        if time_matches:
            timestamps = [[_time_to_seconds(start), _time_to_seconds(end)] for start, end in time_matches]
            break

    if not timestamps:
        for sentence in candidates:
            time_matches = main_pattern.findall(sentence)
            if time_matches:
                timestamps = [[float(start), float(end)] for start, end in time_matches]
                break

    if not timestamps:
        times = []
        for sentence in candidates:
            time = time_format_pattern.findall(sentence)
            if not time:
                continue
            times.extend(_time_to_seconds(timestamp) for timestamp in time)
        times = times[: len(times) // 2 * 2]
        timestamps = [(times[i], times[i + 1]) for i in range(0, len(times), 2)]

    if not timestamps:
        times = []
        for sentence in candidates:
            time = time_number_pattern.findall(sentence)
            if time:
                times.append(float(time[0]))
        times = times[: len(times) // 2 * 2]
        timestamps = [(times[i], times[i + 1]) for i in range(0, len(times), 2)]

    if not timestamps:
        for sentence in candidates:
            fallback_matches = fallback_pattern.findall(sentence)
            if fallback_matches:
                timestamps = [[float(start), float(end)] for start, end in fallback_matches]
                break

    results = []
    for start, end in timestamps[:1]:
        results.append([start, end] if end > start else [end, start])
    return results


def _time_to_seconds(timestamp):
    parts = timestamp.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = map(float, parts)
        return hours * 3600 + minutes * 60 + seconds
    minutes, seconds = map(float, parts)
    return minutes * 60 + seconds


def iou(a, b):
    max0 = max(a[0], b[0])
    min0 = min(a[0], b[0])
    max1 = max(a[1], b[1])
    min1 = min(a[1], b[1])
    denom = max1 - min0
    return 0.0 if denom <= 0 else max(min1 - max0, 0) / denom


def _parse_ground_truth(raw_gt):
    if isinstance(raw_gt, str):
        raw_gt = ast.literal_eval(raw_gt)
    return float(raw_gt[0]), float(raw_gt[1])


def _temporal_grounding_compute_metrics(results):
    combined_submission = {}
    for submission_dict in results:
        combined_submission.update(submission_dict)

    ious = []
    bad_pred = 0
    for key, pred_text in combined_submission.items():
        try:
            gt = _parse_ground_truth(key.rsplit(">>>", 1)[-1])
            pred_times = extract_time(pred_text)
            if len(pred_times) != 1:
                cur_iou = 0.0
                bad_pred += 1
            else:
                cur_iou = iou(gt, pred_times[0])
            ious.append(cur_iou)
        except Exception as e:
            eval_logger.warning(f"Failed to process Charades-STA result: {e}")
            ious.append(0.0)
            bad_pred += 1

    total = len(ious)
    eval_logger.info(f"Charades-STA bad predictions: {bad_pred}/{total}")
    metrics = {}
    for thr in [0.3, 0.5, 0.7]:
        count = sum(1 for value in ious if value >= thr)
        metrics[f"IOU@{int(thr * 10)}"] = count * 100 / total if total else 0
    metrics["mIOU"] = sum(ious) * 100 / total if total else 0
    return metrics


def temporal_grounding_aggregate_iou3(results, args):
    return _temporal_grounding_compute_metrics(results)["IOU@3"]


def temporal_grounding_aggregate_iou5(results, args):
    return _temporal_grounding_compute_metrics(results)["IOU@5"]


def temporal_grounding_aggregate_iou7(results, args):
    return _temporal_grounding_compute_metrics(results)["IOU@7"]


def temporal_grounding_aggregate_miou(results, args):
    return _temporal_grounding_compute_metrics(results)["mIOU"]


def temporal_grounding_aggregate_charades(results, args):
    temporal_grounding_aggregate_submissions(results, args, "charades")


def temporal_grounding_aggregate_submissions(results, args, task):
    now_date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    submission_file_name = f"inference_results_temporal_grounding_{task}_{now_date_time}.json"
    path = file_utils.generate_submission_file(submission_file_name, args)

    # results is a list of 5031 dict,
    # need to convert results into a single dict with 5031 key-value pairs
    combined_submission = {}

    for submission_dict in results:
        combined_submission.update(submission_dict)

    with open(path, "w") as f:
        json.dump(combined_submission, f, indent=4)

    eval_logger.info(f"Submission file saved to {path}")
