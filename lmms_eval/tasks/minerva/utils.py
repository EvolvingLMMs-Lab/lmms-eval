import datetime
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import yaml
from decord import VideoReader, cpu
from lmms_eval.tasks.minerva.download_videos import main as download_videos

import lmms_eval.tasks._task_utils.file_utils as file_utils

with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

# We will unzip all the zip files
# To HF HOME cache dir
# And load it here
HF_HOME = os.environ["HF_HOME"] if "HF_HOME" in os.environ else os.path.expanduser("~/.cache/huggingface/hub")
_CACHE_SUBDIR = config["dataset_kwargs"]["cache_dir"]
CACHE_DIR = Path(os.path.join(HF_HOME, _CACHE_SUBDIR)).expanduser()
VIDEOS_DIR = CACHE_DIR / "videos"
cache_dir = str(CACHE_DIR)
videos_dir = str(VIDEOS_DIR)
OPTIONS = ["A", "B", "C", "D", "E"]

from loguru import logger as eval_logger

from PIL import Image as PIL_Image

import requests
MINERVA_JSON_URL = "https://storage.mtls.cloud.google.com/neptunedata/minerva.json"


def download_file(url: str, target_dir: str | Path, filename: str | None = None) -> Path:
    target_path = Path(target_dir).expanduser().resolve()
    target_path.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = url.split("/")[-1] or "downloaded_file"
    file_path = target_path / filename
    if file_path.exists():
        return file_path

    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        with file_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    return file_path


def download_dataset(
    cache_dir: str | Path | None = None,
    videos_dir: str | Path | None = None,
    url: str = MINERVA_JSON_URL,
    **_,
) -> Path:
    target_cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else CACHE_DIR
    target_videos_dir = Path(videos_dir).expanduser().resolve() if videos_dir else VIDEOS_DIR

    json_file = download_file(url, target_cache_dir, "minerva.json")
    download_videos(output_dir=target_videos_dir, json_file=json_file)
    return json_file

def minerva_doc_to_messages(doc):
    visuals = minerva_doc_to_visual(doc)
    if visuals is None:
        visuals = []
    text = minerva_doc_to_text(doc)
    messages = [{"role": "user", "content": []}]
    content = []
    for visual in visuals:
        if isinstance(visual, PIL_Image.Image):
            content.append({"type": "image", "url": visual})
        elif isinstance(visual, dict):
            content.append({"type": "audio", "url": visual})
        elif isinstance(visual, str):
            content.append({"type": "video", "url": visual, "question": doc['question']})
    content.append({"type": "text", "text": text})
    messages[0]["content"] = content
    return messages


# Pass in video path here
# Can only work correctly with video llm
def minerva_doc_to_visual(doc):
    video_path = doc["video_id"] + ".mp4"
    video_path = os.path.join(videos_dir, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    elif os.path.exists(video_path.replace("mp4", "MP4")):
        video_path = video_path.replace("mp4", "MP4")
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


# This is the place where you format your question
def minerva_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    if "post_prompt" in lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs["post_prompt"]

    question = doc["question"]
    for i, choice in enumerate(OPTIONS):
        question += f"\n{choice}. {doc[f'answer_choice_{i}']}"
    post_prompt = "\nAnswer with the option's letter from the given choices directly."

    return f"{pre_prompt}{question}{post_prompt}"


def minerva_doc_to_answer(doc):
    return doc["answer_id"]


# Process result for mc_ppl
def minerva_process_results(doc, result):
    # Initialize minimum value and index
    min_value = float("inf")
    min_index = -1

    # Iterate through the results to find the index of the lowest value
    for i, (value, _) in enumerate(result):
        if value < min_value:
            min_value = value
            min_index = i

    # Return the result with the index of the lowest value
    return {"submission": {doc["video_id"]: min_index}, "score": {"pred": min_index, "ground_truth": doc["answer_id"]}}


def get_multi_choice_info(doc):
    all_choices = []
    index2ans = {}
    for i in range(len(OPTIONS)):
        # import pdb;pdb.set_trace()
        index2ans[OPTIONS[i]] = doc[f'answer_choice_{i}'].strip()
        all_choices.append(OPTIONS[i])

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
    ans_with_space = False
    ans_with_dot = False
    candidates = []
    # import pdb; pdb.set_trace()
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            candidates.append(f"({choice})")
            ans_with_brack = True

    # if len(candidates) == 0:
    for choice in all_choices:  # e.g., A B C D
        if f"{choice} " in response:
            candidates.append(f"{choice} ")
            ans_with_space = True

    # if len(candidates) == 0:
    for choice in all_choices:  # e.g., A. B. C. D.
        if f"{choice}." in response:
            candidates.append(f"{choice}.")
            ans_with_dot = True

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        # candidates = list(set(candidates))
        start_indexes = []
        if index_ans:
            # if ans_with_brack:
            for can in candidates:
                index = response.rfind(can)
                start_indexes.append(index)  # -1 will be ignored anyway
                # start_indexes = [generated_response.index(f'({can})') for can in candidates]
            # if ans_with_space:
            #     for can in candidates:
            #         index = response.rfind(f"{can} ")
            #         start_indexes.append(index)
            # if ans_with_dot:
            #     for can in candidates:
            #         index = response.rfind(f"{can}.")
            #         start_indexes.append(index)
            # if not ans_with_brack and not ans_with_space and not ans_with_dot:
            #     for can in candidates:
            #         index = response.rfind(f" {can} ")
            #         start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the first one
        pred_index = candidates[np.argmin(start_indexes)]
        pred_index = pred_index.replace("(", "").replace(")", "").replace(".", "").strip()
    else:  # if only one candidate, use it.
        pred_index = candidates[0]
        pred_index = pred_index.replace("(", "").replace(")", "").replace(".", "").strip()

    return pred_index, len(candidates) > 0


# Process result for mcq answer generation
def minerva_process_results_generation(doc, result):
    # import pdb;pdb.set_trace()
    pred = result[0]

    index2ans, all_choices = get_multi_choice_info(doc)
    parsed_pred, matched_tag = parse_multi_choice_response(pred, all_choices, index2ans)

    pred_to_index = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    index = pred_to_index.get(parsed_pred, -1)  # Default to -1 if the prediction is not found

    return {"submission": {doc["video_idx"]: index}, "score": {"pred": index, "ground_truth": doc["answer_id"]}}


def minerva_aggregate_submissions(results, args, task):
    now_date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    submission_file_name = f"inference_results_minerva_{task}_{now_date_time}.json"
    path = file_utils.generate_submission_file(submission_file_name, args)

    # results is a list of 5031 dict,
    # need to convert results into a single dict with 5031 key-value pairs
    combined_submission = {}

    for submission_dict in results:
        combined_submission.update(submission_dict)

    with open(path, "w") as f:
        json.dump(combined_submission, f, indent=4)

    eval_logger.info(f"Submission file saved to {path}")


# Factory into different aggregate
def minerva_aggregate_mc(results, args):
    minerva_aggregate_submissions(results, args, "MC")


def minerva_aggregate_mc_ppl(results, args):
    minerva_aggregate_submissions(results, args, "MC_PPL")


def minerva_aggregate_score(results, args):
    yes_count = 0

    # results is a list of dict
    for answer_dict in results:
        if str(answer_dict["ground_truth"]) == str(answer_dict["pred"]):
            yes_count = yes_count + 1

    accuracy = yes_count / len(results)

    return accuracy


def minerva_doc_to_choice(doc):
    return [doc[f'answer_choice_{i}'] for i in range(len(OPTIONS))]
