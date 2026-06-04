import json
import os
import re
from pathlib import Path

import yaml
from loguru import logger as eval_logger

hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)

with open(Path(__file__).parent / "_default_yaml_template", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
    config = yaml.safe_load("".join(safe_data))
    cache_name = config["dataset_kwargs"]["cache_dir"]


video_cache_dir = os.path.join(base_cache_dir, cache_name)


def extract_time(paragraph: str):
    paragraph = paragraph.lower()
    timestamps = []

    time_regex = re.compile(r"\b(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?|\d{1,2}:\d{2}(?:\.\d+)?)\b")
    time_matches = re.findall(time_regex, paragraph)
    time_matches = time_matches[: len(time_matches) // 2 * 2]

    if time_matches:
        conv = []
        for t in time_matches:
            parts = t.split(":")
            if len(parts) == 3:
                h, m = map(int, parts[:2])
                s = float(parts[2])
                conv.append(h * 3600 + m * 60 + s)
            else:
                m = int(parts[0])
                s = float(parts[1])
                conv.append(m * 60 + s)
        timestamps = [(conv[i], conv[i + 1]) for i in range(0, len(conv), 2)]

    if not timestamps:
        patterns = [
            r"(\d+\.?\d*)\s*-\s*(\d+\.?\d*)",
            r"(\d+\.?\d*)\s+to\s+(\d+\.?\d*)",
        ]
        for p in patterns:
            matches = re.findall(p, paragraph)
            if matches:
                timestamps = [(float(s), float(e)) for s, e in matches]
                break

    if not timestamps:
        nums = re.findall(r"\b\d+\.?\d*\b", paragraph)
        nums = nums[: len(nums) // 2 * 2]
        timestamps = [(float(nums[i]), float(nums[i + 1])) for i in range(0, len(nums), 2)]

    return timestamps


def iou(a, b):
    max0 = max(a[0], b[0])
    min0 = min(a[0], b[0])
    max1 = max(a[1], b[1])
    min1 = min(a[1], b[1])
    denom = max1 - min0
    return 0.0 if denom <= 0 else max(min1 - max0, 0) / denom


def parse_target_from_key(k: str):
    try:
        target_str = k.rsplit(">>>", 1)[-1]
        tgt = json.loads(target_str)
        return float(tgt[0][0]), float(tgt[0][1])
    except Exception as e:
        raise ValueError(f"Failed to parse target from key: {k}") from e


def timelens_doc_to_visual(doc):
    video_path = doc["video_path"]

    full_video_path = os.path.join(video_cache_dir, video_path)

    if os.path.exists(full_video_path):
        return [full_video_path]
    else:
        eval_logger.warning(f"Video not found: {full_video_path}")
        return [full_video_path]


def timelens_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    query = doc["query"]

    return f"{pre_prompt}{query}{post_prompt}"


def timelens_doc_to_target(doc):
    return doc["span"]


TIMELENS_METRICS = ["IOU@3", "IOU@5", "IOU@7", "mIOU"]


def timelens_process_results(doc, result):
    pred = result[0]

    key = f'{doc["video_path"]}>>>{doc["query"]}>>>{doc["span"]}'

    data_dict = {key: pred}

    return {f"timelens_{metric}": data_dict for metric in TIMELENS_METRICS}


def _timelens_compute_metrics(results, args=None):

    combined_submission = {}
    for submission_dict in results:
        combined_submission.update(submission_dict)

    ious = []
    num_annos = 0
    bad_pred = 0

    for k, pred_text in combined_submission.items():
        try:
            target_str = k.rsplit(">>>", 1)[-1]
            tgt = json.loads(target_str)
            gt = (float(tgt[0][0]), float(tgt[0][1]))

            pred_times = extract_time(pred_text)
            if not pred_times:
                cur_iou = 0.0
                bad_pred += 1
            else:
                pred = pred_times[0]
                cur_iou = iou(pred, gt)

            ious.append(cur_iou)
            num_annos += 1
        except Exception as e:
            eval_logger.warning(f"Failed to process result: {e}")

    metrics = {}
    for thr in [0.3, 0.5, 0.7]:
        count = sum(1 for v in ious if v >= thr)
        metrics[f"IOU@{int(thr*10)}"] = count * 100 / num_annos if num_annos > 0 else 0

    metrics["mIOU"] = sum(ious) * 100 / num_annos if num_annos > 0 else 0

    # eval_logger.info(
    #     f"Num samples: {num_annos}\n"
    #     f"Bad pred (no time parsed): {bad_pred}\n"
    #     f"IOU@3: {metrics['IOU@3']:.2f}\n"
    #     f"IOU@5: {metrics['IOU@5']:.2f}\n"
    #     f"IOU@7: {metrics['IOU@7']:.2f}\n"
    #     f"mIOU: {metrics['mIOU']:.2f}"
    # )

    return metrics


def timelens_aggregate_iou3(results, args):
    metrics = _timelens_compute_metrics(results, args)
    return metrics["IOU@3"]


def timelens_aggregate_iou5(results, args):
    metrics = _timelens_compute_metrics(results, args)
    return metrics["IOU@5"]


def timelens_aggregate_iou7(results, args):
    metrics = _timelens_compute_metrics(results, args)
    return metrics["IOU@7"]


def timelens_aggregate_miou(results, args):
    metrics = _timelens_compute_metrics(results, args)
    return metrics["mIOU"]
