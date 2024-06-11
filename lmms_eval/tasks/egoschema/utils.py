from decord import VideoReader, cpu
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

# We will unzip all the zip files
# To HF HOME cache dir
# And load it here
HF_HOME = os.environ["HF_HOME"]
cache_dir = config["dataset_kwargs"]["cache_dir"]
cache_dir = os.path.join(HF_HOME, cache_dir)
cache_dir = os.path.join(cache_dir, "videos")

eval_logger = logging.getLogger("lmms-eval")


# Pass in video path here
# Can only work correctly with video llm
def egoschema_doc_to_visual(doc):
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
def egoschema_doc_to_text(doc, model_specific_prompt_kwargs=None):
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


def egoschema_doc_to_answer(doc):
    return doc["answer"]


# Process result for mc_ppl
def egoschema_process_results(doc, result):
    # Initialize minimum value and index
    min_value = float("inf")
    min_index = -1

    # Iterate through the results to find the index of the lowest value
    for i, (value, _) in enumerate(result):
        if value < min_value:
            min_value = value
            min_index = i

    # Return the result with the index of the lowest value
    return {"submission": {doc["video_idx"]: min_index}, "score": {"pred": min_index, "ground_truth": doc["answer"]}}


# Process result for mcq answer generation
def egoschema_process_results_generation(doc, result):
    pred = result[0]
    
    # Determine whether the video LLM output is correct, based on word matching rules
    # Ensure each option string ends with a period
    option_strs = [opt if opt.endswith(".") else opt + "." for opt in doc["option"]]  # Complete option strings
    option_sents = [opt.split(". ")[1] if ". " in opt else opt for opt in option_strs]  # Option sentence
    option_inds = [opt.split(". ")[0] if ". " in opt else opt for opt in option_strs]  # Option letter, e.g., A, B, C, D, E
    
    video_llm_pred = None
    index = -1
    
    # Check if the prediction matches any of the complete option strings
    for idx, option_str in enumerate(option_strs):
        if pred == option_str:
            video_llm_pred = option_str
            index = idx
            break
    
    # Check if the prediction matches any of the option sentences
    if not video_llm_pred:
        for idx, option_sent in enumerate(option_sents):
            if pred == option_sent:
                video_llm_pred = option_sent
                index = idx
                break

    # Check if the prediction matches any of the option letters
    if not video_llm_pred:
        for idx, option_ind in enumerate(option_inds):
            if pred == option_ind or pred == option_ind.replace(".", ""):
                video_llm_pred = option_ind
                index = idx
                break
            
    return {"submission": {doc["video_idx"]: index}, "score": {"pred": index, "ground_truth": doc["answer"]}}


def egoschema_aggregate_submissions(results, args, task):
    now_date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    submission_file_name = f"inference_results_egoschema_{task}_{now_date_time}.json"
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
def egoschema_aggregate_mc(results, args):
    egoschema_aggregate_submissions(results, args, "MC")


def egoschema_aggregate_mc_ppl(results, args):
    egoschema_aggregate_submissions(results, args, "MC_PPL")


def egoschema_aggregate_score(results, args):
    yes_count = 0

    # results is a list of dict
    for answer_dict in results:
        if str(answer_dict["ground_truth"]) == str(answer_dict["pred"]):
            yes_count = yes_count + 1

    accuracy = yes_count / len(results)

    return accuracy


def egoschema_doc_to_choice(doc):
    return [op.split(".")[1].strip() for op in doc["option"]]
