import numpy as np
import os
import sys
import datetime
import lmms_eval.tasks._task_utils.file_utils as file_utils
import json
import logging
import yaml
from pathlib import Path

with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

# A bit ugly here
# But the idea is that we will unzip all the zip files
# To HF HOME cache dir
# And load it here
HF_HOME = os.getenv("HF_HOME", "~/.cache/huggingface/")
cache_dir = config["dataset_kwargs"]["cache_dir"]
cache_dir = os.path.join(HF_HOME, cache_dir)
cache_dir = os.path.join(cache_dir, "videos")


eval_logger = logging.getLogger("lmms-eval")


# Pass in video path here
# Can only work correctly with video llm
def worldqa_doc_to_visual(doc):
    video_path = doc["video_idx"] + ".mp4"
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    elif os.path.exists(video_path.replace("mp4", "MP4")):
        video_path = video_path.replace("mp4", "MP4")
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


# This is the place where you format your question
def worldqa_doc_to_text(doc, model_specific_prompt_kwargs=None):
    if model_specific_prompt_kwargs is None:
        model_specific_prompt_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in model_specific_prompt_kwargs:
        pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
    if "post_prompt" in model_specific_prompt_kwargs:
        post_prompt = model_specific_prompt_kwargs["post_prompt"]

    question = doc["question"]
    if "option" in doc:
        for op in doc["option"]:
            question += "\n" + op
        post_prompt = "\nAnswer with the option's letter from the given choices directly."

    return f"{pre_prompt}{question}{post_prompt}"


def worldqa_doc_to_answer(doc):
    return doc["answer"]


# If it is mc, keep the option for exact match
def worldqa_doc_to_answer_mc(doc):
    return doc["answer"].split(".")[0].strip()


# If it is mc ppl, keep the option str for perplexity base matching
def worldqa_doc_to_answer_mc_ppl(doc):
    return doc["answer"].split(".")[1].strip()


# An example of showing how to custom metric
# Your metric name should have the same key name in your return dict
def worldqa_process_results(doc, result):
    pred = result[0]
    return {"submission": {"pred": pred, "question_idx": doc["question_idx"], "object_description": doc["object_description"], "answer": doc["answer"]}}


def worldqa_aggregate_submissions(results, args, task):
    now_date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    submission_file_name = f"worldqa-{task}-{now_date_time}.json"
    path = file_utils.generate_submission_file(submission_file_name, args)
    with open(path, "w") as f:
        json.dump(results, f)
    eval_logger.info(f"Submission file saved to {path}")


# Factory into different aggregate
def worldqa_aggregate_gen(results, args):
    worldqa_aggregate_submissions(results, args, "Generation")


def worldqa_aggregate_mc(results, args):
    worldqa_aggregate_submissions(results, args, "MC")


def worldqa_aggregate_mc_ppl(results, args):
    worldqa_aggregate_submissions(results, args, "MC_PPL")


def worldqa_doc_to_choice(doc):
    return [op.split(".")[1].strip() for op in doc["option"]]
