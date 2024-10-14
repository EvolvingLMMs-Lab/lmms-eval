import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import decord
import numpy as np
import torch
import yaml
from decord import VideoReader, cpu
from loguru import logger as eval_logger
from PIL import Image

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file


def timestamp_to_seconds(timestamp):
    # Split the timestamp into hours, minutes, and seconds
    h, m, s = timestamp.split(":")
    # Convert hours, minutes, and total seconds (including fractions) to float and compute total seconds
    total_seconds = int(h) * 3600 + int(m) * 60 + float(s)
    return total_seconds


def load_video(video_file, duration, max_num_frames=16):
    from decord import VideoReader

    vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
    fps = vr.get_avg_fps()
    total_valid_frames = int(duration * fps)
    num_frames = min(max_num_frames, int(duration))

    frame_indices = [int(total_valid_frames / num_frames) * i for i in range(num_frames)]

    frames = vr.get_batch(frame_indices)
    if isinstance(frames, torch.Tensor):
        frames = frames.numpy()
    else:
        frames = frames.asnumpy()
    frame_timestamps = [frame_index / fps for frame_index in frame_indices]

    return [Image.fromarray(fr).convert("RGB") for fr in frames]


def compute_frame_timestamps(duration, max_num_frames=16):
    if duration > max_num_frames:
        return [duration / max_num_frames * i for i in range(max_num_frames)]
    else:
        return [i for i in range(int(duration))]


def insert_subtitles_into_frames(frame_timestamps, subtitles, starting_timestamp_for_subtitles, duration):
    interleaved_list = []
    cur_i = 0

    for subtitle in subtitles:
        if "timestamp" in subtitle:
            start, end = subtitle["timestamp"]

            if not isinstance(end, float):
                end = duration

            start -= starting_timestamp_for_subtitles
            end -= starting_timestamp_for_subtitles

            subtitle_timestamp = (start + end) / 2
            subtitle_text = subtitle["text"]
        else:
            start, end = subtitle["start"], subtitle["end"]
            start = timestamp_to_seconds(start)
            end = timestamp_to_seconds(end)
            start -= starting_timestamp_for_subtitles
            end -= starting_timestamp_for_subtitles

            subtitle_timestamp = (start + end) / 2
            subtitle_text = subtitle["line"]

        for i, frame_timestamp in enumerate(frame_timestamps[cur_i:]):
            if frame_timestamp <= subtitle_timestamp:
                # print("frame:", frame_timestamp)
                interleaved_list.append("<image>")
                cur_i += 1
            else:
                break

        if end - start < 1:
            end = subtitle_timestamp + 0.5
            start = subtitle_timestamp - 0.5

        covering_frames = False
        for frame_timestamp in frame_timestamps:
            if frame_timestamp < end and frame_timestamp > start:
                covering_frames = True
                break

        if covering_frames:
            # print("subtitle:", subtitle_timestamp, start, end)
            interleaved_list.append(subtitle_text)
        else:
            pass
            # print("leaving out subtitle:", start, end)

    for i, frame_timestamp in enumerate(frame_timestamps[cur_i:]):
        # print(frame_timestamp)
        interleaved_list.append("<image>")

    return "\n".join(interleaved_list)


def longvideobench_doc_to_text(doc, lmms_eval_specific_kwargs):
    candidates = []

    for i in range(5):
        candidate = doc.get(f"option{i}")
        if candidate != "N/A":
            candidates.append(candidate)

    question = doc["question"] + "\n" + "\n".join([". ".join([chr(ord("A") + i), candidate]) for i, candidate in enumerate(candidates)])
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]

    if lmms_eval_specific_kwargs.get("insert_interleave_subtitles", False):
        with open(Path(__file__).parent / "longvideobench_val_i.yaml", "r") as f:
            raw_data = f.readlines()
            safe_data = []
            for i, line in enumerate(raw_data):
                # remove function definition since yaml load cannot handle it
                if "!function" not in line:
                    safe_data.append(line)
        cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]
        subtitle_subdir_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"].get("subtitle_subdir", "subtitles")
        cache_dir = os.path.join(base_cache_dir, cache_name, subtitle_subdir_name)
        with open(os.path.join(cache_dir, doc["subtitle_path"])) as f:
            subtitles = json.load(f)

        max_num_frames = yaml.safe_load("".join(safe_data))["dataset_kwargs"].get("max_num_frames", 16)

        frame_timestamps = compute_frame_timestamps(doc["duration"], max_num_frames)
        interleaved_prefix = insert_subtitles_into_frames(frame_timestamps, subtitles, doc["starting_timestamp_for_subtitles"], doc["duration"])
        return f"{pre_prompt}{interleaved_prefix}\n{question}\n{post_prompt}"
    else:
        return f"{pre_prompt}{question}\n{post_prompt}"


hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)


def longvideobench_doc_to_visual_v(doc):
    with open(Path(__file__).parent / "longvideobench_val_v.yaml", "r") as f:
        raw_data = f.readlines()
        safe_data = []
        for i, line in enumerate(raw_data):
            # remove function definition since yaml load cannot handle it
            if "!function" not in line:
                safe_data.append(line)
    cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]
    vid_subdir_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"].get("video_subdir", "videos/")
    cache_dir = os.path.join(base_cache_dir, cache_name, vid_subdir_name)
    video_path = doc["video_path"]
    video_path = os.path.join(cache_dir, video_path)
    return [video_path]


def longvideobench_doc_to_visual_i(doc):
    with open(Path(__file__).parent / "longvideobench_val_i.yaml", "r") as f:
        raw_data = f.readlines()
        safe_data = []
        for i, line in enumerate(raw_data):
            # remove function definition since yaml load cannot handle it
            if "!function" not in line:
                safe_data.append(line)
    cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]
    vid_subdir_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"].get("video_subdir", "videos/")
    cache_dir = os.path.join(base_cache_dir, cache_name, vid_subdir_name)
    video_path = doc["video_path"]
    video_path = os.path.join(cache_dir, video_path)
    max_num_frames = yaml.safe_load("".join(safe_data))["dataset_kwargs"].get("max_num_frames", 16)
    return load_video(video_path, doc["duration"], max_num_frames)


def get_multi_choice_info(options):
    """
    Given the list of options for multiple choice question
    Return the index2ans and all_choices
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/data_utils.py#L54
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
    Return the predicted index e.g., A, B, C, D.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L10
    """
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f"{choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A. B. C. D.
            if f"{choice}." in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)  # -1 will be ignored anyway
                # start_indexes = [generated_response.index(f'({can})') for can in candidates]
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index


def evaluate_longvideobench(samples):
    pred_correct = 0
    judge_dict = dict()
    for sample in samples:
        gold_i = sample["answer"]
        pred_i = sample["parsed_pred"]
        correct = eval_multi_choice(gold_i, pred_i)

        if correct:
            judge_dict[sample["id"]] = "Correct"
            pred_correct += 1
        else:
            judge_dict[sample["id"]] = "Wrong"

    if len(samples) == 0:
        return {"acc": 0}
    return judge_dict, {"acc": pred_correct / len(samples)}


def eval_multi_choice(gold_i, pred_i):
    correct = False
    # only they are exactly the same, we consider it as correct
    if isinstance(gold_i, list):
        for answer in gold_i:
            if answer == pred_i:
                correct = True
                break
    else:  # gold_i is a string
        if gold_i == pred_i:
            correct = True
    return correct


def calculate_ins_level_acc(results):
    """Calculate the instruction level accuracy for given Subject results
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L246
    """
    acc = 0
    ins_num = 0
    for cat_results in results.values():
        acc += cat_results["acc"] * cat_results["num_example"]
        ins_num += cat_results["num_example"]
    if ins_num == 0:
        return 0
    return acc / ins_num


def longvideobench_process_results(doc, results):
    pred = results[0]
    all_choices = []
    index2ans = {}
    for i in range(5):
        option = doc.get(f"option{i}")
        if option == "N/A":
            break
        index2ans[chr(ord("A") + i)] = option
        all_choices.append(chr(ord("A") + i))

    parsed_pred = parse_multi_choice_response(pred, all_choices, index2ans)
    id = doc["video_id"]
    lvb_acc = {"id": id, "duration_group": doc["duration_group"], "question_category": doc["question_category"], "answer": chr(ord("A") + doc["correct_choice"]), "parsed_pred": parsed_pred}
    return {
        "lvb_acc": lvb_acc,
        "submission": {
            id: pred,
        },
    }


def longvideobench_aggregate_results(results):
    evaluation_result = {}
    subset_to_eval_samples = defaultdict(list)
    for result in results:
        subset_to_eval_samples[result["duration_group"]].append(result)
        subset_to_eval_samples[result["question_category"]].append(result)
    for subset, sub_eval_samples in subset_to_eval_samples.items():
        judge_dict, metric_dict = evaluate_longvideobench(sub_eval_samples)
        metric_dict.update({"num_example": len(sub_eval_samples)})
        evaluation_result[subset] = metric_dict
    printable_results = {}

    for cat_name, cat_results in evaluation_result.items():
        printable_results[cat_name] = {
            "num": int(cat_results["num_example"]),
            "acc": round(cat_results["acc"], 5),
        }
    all_ins_acc = calculate_ins_level_acc(evaluation_result)
    printable_results["Overall"] = {
        "num": sum([cat_results["num_example"] for cat_results in evaluation_result.values()]),
        "acc": round(all_ins_acc, 5),
    }
    eval_logger.info(printable_results)
    return printable_results["Overall"]["acc"]


def longvideobench_aggregate_results_for_submission(results, args):
    path = generate_submission_file("longvideobench_test_for_submission.json", args)
    results_dict = {list(item.keys())[0]: list(item.values())[0] for item in results}
    with open(path, "w") as f:
        json.dump(results_dict, f)
    eval_logger.info(f"Results saved to {path}.")
