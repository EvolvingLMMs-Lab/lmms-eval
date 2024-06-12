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

import requests
import openai
from openai import OpenAI
import time
import ast
from tqdm import tqdm
import random

import re

with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))


API_TYPE = os.getenv("API_TYPE", "openai")

if API_TYPE == "openai":
    API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

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
def tempcompass_doc_to_visual(doc):
    video_path = doc["video_id"] + ".mp4"
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


# This is the place where you format your question
def tempcompass_doc_to_text_multi_choice(doc, model_specific_prompt_kwargs=None):
    if model_specific_prompt_kwargs is None:
        model_specific_prompt_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in model_specific_prompt_kwargs:
        pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
    if "post_prompt" in model_specific_prompt_kwargs:
        post_prompt = model_specific_prompt_kwargs["post_prompt"]["multi-choice"]

    question = doc["question"]
    return f"{pre_prompt}{question}{post_prompt}"


def tempcompass_doc_to_text_yes_no(doc, model_specific_prompt_kwargs=None):
    if model_specific_prompt_kwargs is None:
        model_specific_prompt_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in model_specific_prompt_kwargs:
        pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
    if "post_prompt" in model_specific_prompt_kwargs:
        post_prompt = model_specific_prompt_kwargs["post_prompt"]["yes_no"]

    question = doc["question"]
    return f"{pre_prompt}{question}{post_prompt}"


def tempcompass_doc_to_text_caption_matching(doc, model_specific_prompt_kwargs=None):
    if model_specific_prompt_kwargs is None:
        model_specific_prompt_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in model_specific_prompt_kwargs:
        pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
    if "post_prompt" in model_specific_prompt_kwargs:
        post_prompt = model_specific_prompt_kwargs["post_prompt"]["caption_matching"]

    question = doc["question"]
    return f"{pre_prompt}{question}{post_prompt}"


def tempcompass_doc_to_text_captioning(doc, model_specific_prompt_kwargs=None):
    if model_specific_prompt_kwargs is None:
        model_specific_prompt_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in model_specific_prompt_kwargs:
        pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
    if "post_prompt" in model_specific_prompt_kwargs:
        post_prompt = model_specific_prompt_kwargs["post_prompt"]["captioning"]

    question = doc["question"]
    return f"{pre_prompt}{question}{post_prompt}"


def tempcompass_doc_to_answer(doc):
    return doc["answer"]


# Process result for multi_choice
def tempcompass_process_results_multi_choice(doc, result):
    pred = result[0]
    rating = 0
    match_success = True
    chatgpt_response = None

    # Some hand-crafted matching rules
    if pred == doc["answer"]:
        rating = 1
    elif pred in ["A", "B", "C", "D"]:
        rating = 1 if pred == doc["answer"][0] else 0
    elif any(pred.startswith(prefix) for prefix in ["A.", "B.", "C.", "D."]):
        rating = 1 if pred.split(".")[0] == doc["answer"][0] else 0
    elif any(pred.startswith(prefix) for prefix in ["A)", "B)", "C)", "D)"]):
        rating = 1 if pred.split(")")[0] == doc["answer"][0] else 0
    else:
        # Fail to match answer in the video-llm response. Use ChatGPT to evaluate.
        match_success = False

        base_prompt = """
        You will receive a multi-choice question, the ground-truth answer and the prediction from a question answering (QA) model. \
        Your task is to determine whether QA model prediction is correct, based on the question and ground-truth answer. \
        If the prediction is correct, respond "Correct". If the prediction is incorrect, respond "Incorrect".
        """
        prompt = f"""{base_prompt}\nMulti-Choice Question:\n{doc["question"]}\nGround-Truth Answer: {doc["answer"]}\nModel Prediction: {pred}"""
        chatgpt_response, rating = get_eval_result(prompt)

    if chatgpt_response:
        return {
            "avg_accuracy": {
                "video_id": doc["video_id"],
                "question": doc["question"],
                "gt-answer": doc["answer"],
                "video-llm-prediction": pred,
                "match_success": match_success,
                "rating": rating,
                "chatgpt_response": chatgpt_response,
                "dim": doc["dim"],
            },
            doc["dim"]
            + "_accuracy": {
                "video_id": doc["video_id"],
                "question": doc["question"],
                "gt-answer": doc["answer"],
                "video-llm-prediction": pred,
                "match_success": match_success,
                "rating": rating,
                "chatgpt_response": chatgpt_response,
                "dim": doc["dim"],
            },
        }
    else:
        return {
            "avg_accuracy": {"video_id": doc["video_id"], "question": doc["question"], "gt-answer": doc["answer"], "video-llm-prediction": pred, "match_success": match_success, "rating": rating, "dim": doc["dim"]},
            doc["dim"] + "_accuracy": {"video_id": doc["video_id"], "question": doc["question"], "gt-answer": doc["answer"], "video-llm-prediction": pred, "match_success": match_success, "rating": rating, "dim": doc["dim"]},
        }


# Process result for yes_no
def tempcompass_process_results_yes_no(doc, result):
    pred = result[0]
    rating = 0
    match_success = True
    chatgpt_response = None

    yes_no_pred = extract_pred(pred)

    # Some hand-crafted matching rules
    if yes_no_pred:
        rating = 1 if yes_no_pred == doc["answer"] else 0
    else:
        match_success = False  # Fail to match answer in the video-llm response. Use ChatGPT to evaluate.
        base_prompt = """
        You will receive a Yes/No question, the ground-truth answer and the prediction from a question answering (QA) model. \
        Your task is to determine whether QA model prediction is correct, based on the question and ground-truth answer. \
        If the prediction is correct, respond "Correct". If the prediction is incorrect, respond "Incorrect".
        """
        prompt = f"""{base_prompt}\nYes/No Question:\n{doc["question"]}\nGround-Truth Answer: {doc["answer"]}\nModel Prediction: {pred}"""
        chatgpt_response, rating = get_eval_result(prompt)

    if chatgpt_response:
        return {
            "avg_accuracy": {
                "video_id": doc["video_id"],
                "question": doc["question"],
                "gt-answer": doc["answer"],
                "video-llm-prediction": pred,
                "match_success": match_success,
                "rating": rating,
                "chatgpt_response": chatgpt_response,
                "dim": doc["dim"],
            },
            doc["dim"]
            + "_accuracy": {
                "video_id": doc["video_id"],
                "question": doc["question"],
                "gt-answer": doc["answer"],
                "video-llm-prediction": pred,
                "match_success": match_success,
                "rating": rating,
                "chatgpt_response": chatgpt_response,
                "dim": doc["dim"],
            },
        }
    else:
        return {
            "avg_accuracy": {"video_id": doc["video_id"], "question": doc["question"], "gt-answer": doc["answer"], "video-llm-prediction": pred, "match_success": match_success, "rating": rating, "dim": doc["dim"]},
            doc["dim"] + "_accuracy": {"video_id": doc["video_id"], "question": doc["question"], "gt-answer": doc["answer"], "video-llm-prediction": pred, "match_success": match_success, "rating": rating, "dim": doc["dim"]},
        }


# Process result for caption_matching
def tempcompass_process_results_caption_matching(doc, result):
    pred = result[0]
    rating = 0
    match_success = True
    chatgpt_response = None

    eval_rule_rating = eval_rule(pred, doc["question"], doc["answer"])

    # Some hand-crafted matching rules
    if eval_rule_rating != "fail":
        rating = eval_rule_rating
    else:
        match_success = False  # Fail to match answer in the video-llm response. Use ChatGPT to evaluate.
        base_prompt = """
        You will receive a caption matching question, the ground-truth answer and the prediction from a question answering (QA) model. \
        Your task is to determine whether QA model prediction is correct, based on the question and ground-truth answer. \
        If the prediction is correct, respond "Correct". If the prediction is incorrect, respond "Incorrect".
        """
        prompt = f"""{base_prompt}\nCaption Matching Question:\n{doc["question"]}\nGround-Truth Answer: {doc["answer"]}\nModel Prediction: {pred}"""
        chatgpt_response, rating = get_eval_result(prompt)

    if chatgpt_response:
        return {
            "avg_accuracy": {
                "video_id": doc["video_id"],
                "question": doc["question"],
                "gt-answer": doc["answer"],
                "video-llm-prediction": pred,
                "match_success": match_success,
                "rating": rating,
                "chatgpt_response": chatgpt_response,
                "dim": doc["dim"],
            },
            doc["dim"]
            + "_accuracy": {
                "video_id": doc["video_id"],
                "question": doc["question"],
                "gt-answer": doc["answer"],
                "video-llm-prediction": pred,
                "match_success": match_success,
                "rating": rating,
                "chatgpt_response": chatgpt_response,
                "dim": doc["dim"],
            },
        }
    else:
        return {
            "avg_accuracy": {"video_id": doc["video_id"], "question": doc["question"], "gt-answer": doc["answer"], "video-llm-prediction": pred, "match_success": match_success, "rating": rating, "dim": doc["dim"]},
            doc["dim"] + "_accuracy": {"video_id": doc["video_id"], "question": doc["question"], "gt-answer": doc["answer"], "video-llm-prediction": pred, "match_success": match_success, "rating": rating, "dim": doc["dim"]},
        }


# Process result for captioning
def tempcompass_process_results_captioning(doc, result):
    pred = result[0]

    caption_evaluation_prompt = """
    You will receive a video description and a multi-choice question. Your task is to choose the correct answer and briefly explain the reason why you choose the answer. \
    If none of the choice candidates are correct or the video description lacks enough information to answer the question, just answer "None of the choices are correct". \
    Please organize your response in this format:
    ```
    Reasoning: [Your reason to obtain the answer]
    Answer: [Your answer]
    ```

    Here are some examples of video description, multi-choice question and the expected answer:
    ```
    Video Description: A person is palying football.
    Multi-Choice Question:
    What is the person doing in the video?
    A. cooking
    B. palying football
    C. playing basketball
    D. reading book
    Reasoning: The video description mentions that the person is playing football.
    Answer: B. palying football

    Video Description: A bird is flying clockwise.
    Multi-Choice Question:
    In which direction is the bird flying?
    A. backwark
    B. counter-clockwise
    C. clockwise
    D. downward
    Reasoning: The video description mentions that the bird is flying clockwise
    Answer: C. clockwise

    Video Description: An air balloon is inflating.
    Multi-Choice Question:
    What is happening to the air balloon?
    A. exploding
    B. getting smaller
    C. flying
    Reasoning: The video description mentions that the air balloon is inflating, while none of the coices can be explained as inflating.
    Answer: None of the choices are correct
    ```
    """

    prompt = f"""{caption_evaluation_prompt}\nVideo Description:{pred}\nMulti-Choice Question:\n{doc["mc_question"]}\nAnswer:"""
    eval_result = get_eval_result_for_captioning(prompt, mc_answer=doc["mc_answer"])

    return {
        "avg_accuracy": {
            "video_id": doc["video_id"],
            "question": doc["question"],
            "chatgpt-reasoning": eval_result["chatgpt-reasoning"],
            "chatgpt-answer": eval_result["chatgpt-answer"],
            "video-llm-prediction": pred,
            "gt-answer": doc["mc_answer"],
            "rating": eval_result["rating"],
            "dim": doc["dim"],
        },
        doc["dim"]
        + "_accuracy": {
            "video_id": doc["video_id"],
            "question": doc["question"],
            "chatgpt-reasoning": eval_result["chatgpt-reasoning"],
            "chatgpt-answer": eval_result["chatgpt-answer"],
            "video-llm-prediction": pred,
            "gt-answer": doc["mc_answer"],
            "rating": eval_result["rating"],
            "dim": doc["dim"],
        },
    }


# utils functions for captioning: parse gpt outputs
def parse_llm_output_for_captioning(llm_output, gt_answer):

    if llm_output == "invalid_request_error" or not llm_output:
        eval_result = {"rating": -1, "chatgpt-answer": None, "chatgpt-reasoning": None}
        return eval_result

    eval_result = {}
    lines = llm_output.split("\n")

    for line in lines:
        line = line.strip()
        if "Reasoning" in line:
            eval_result["chatgpt-reasoning"] = line.replace("Reasoning:", "").strip()
        if "Answer" in line:
            eval_result["chatgpt-answer"] = line.replace("Answer:", "").strip()

    if not "chatgpt-answer" in eval_result:
        eval_result["chatgpt-answer"] = llm_output
    if not "chatgpt-reasoning" in eval_result:
        eval_result["chatgpt-reasoning"] = None

    # Check if the chatgpt answer is the ground-truth answer
    answer_counts = sum(eval_result["chatgpt-answer"].count(prefix) for prefix in ["A.", "B.", "C.", "D."])  # calculate the number of 'A.', 'B.', 'C.', 'D.' in chatgpt-answer

    if eval_result["chatgpt-answer"].split(". ")[0] == gt_answer.split(". ")[0] and answer_counts == 1:
        eval_result["rating"] = 1
    else:
        eval_result["rating"] = 0
    return eval_result


# utils functions for captioning: get gpt outputs
def get_llm_output_for_captioning(prompt):
    data = {
        "max_tokens": 128,
        "model": "gpt-3.5-turbo-1106",
        "temperature": 1.0,
        "top_p": 1,
        "presence_penalty": 1,
        "messages": [{"role": "system", "content": "You are an AI assistant for question answering."}, {"role": "user", "content": prompt}],
    }
    response = requests.post(API_URL, headers=headers, data=json.dumps(data).encode("utf-8"))
    result = response.content.decode("utf-8")
    dict_result = json.loads(result)
    token_count = dict_result["usage"]
    try:
        llm_output = dict_result["choices"][0]["message"]["content"].strip()
    except:
        if "error" in dict_result and dict_result["error"]["type"] == "invalid_request_error":
            llm_output = "invalid_request_error"
        else:
            llm_output = ""
    return llm_output, token_count


# utils functions for captioning: consolidate and return gpt outputs
def get_eval_result_for_captioning(prompt, mc_answer, maxtry=10):
    while True:
        try:
            llm_output, token_count = get_llm_output_for_captioning(prompt)
            eval_result = parse_llm_output_for_captioning(llm_output, gt_answer=mc_answer)
            eval_result["token_count"] = token_count
            return eval_result
        except:
            if maxtry <= 0:
                eval_result = {"chatgpt-reasoning": None, "chatgpt-answer": None, "rating": -1, "token_count": None}
                return eval_result
            maxtry -= 1
            print(f"Not success! {maxtry} retries remaining...")
            time.sleep(random.uniform(1, 2))


# utils function for caption_matching
def eval_rule(video_llm_output, question, answer):
    # Determine whether the video llm output is correct, based on word matching rules
    option_strs = question.split("\n")[1:]  # complete option strings
    option_sents = [opt.split(": ")[1] for opt in option_strs]  # option sentence
    option_inds = [opt.split(": ")[0] for opt in option_strs] + [opt.split(": ")[0].replace("Sentence ", "").replace("Option ", "").replace("Caption ", "") for opt in option_strs]  # option index, e.g., Sentence A, Caption A, Option 1
    video_llm_pred = None
    for option_str in option_strs:
        if option_str == video_llm_output:
            video_llm_pred = option_str
    for option_sent in option_sents:
        if option_sent == video_llm_output or (") " in video_llm_output and option_sent == video_llm_output.split(") ")[1]):
            video_llm_pred = option_sent
    for option_ind in option_inds:
        if option_ind == video_llm_output or option_ind == video_llm_output.replace(".", ""):
            video_llm_pred = option_ind

    if video_llm_pred is None:
        return "fail"
    else:
        return 1 if video_llm_pred == answer or video_llm_pred == answer.split(":")[0] or video_llm_pred == answer.split(": ")[1] or video_llm_pred == answer.split(": ")[0].split()[1] else 0


# utils function for yes_no
def extract_pred(video_llm_output):

    # Extract the yes/no predction from the original video llm output
    video_llm_output = video_llm_output.lower()
    if video_llm_output.startswith("yes"):
        return "yes"
    elif video_llm_output.startswith("no"):
        return "no"
    else:
        return False


# utils function for gpt_evaluation when rule-based matching is unsuccessful
def get_eval_result(prompt, maxtry=10, sys_prompt=None):
    llm_output = None
    while True:
        try:
            llm_output = get_llm_output(prompt, sys_prompt)
            rating = llm_output_to_rating(llm_output)
            return llm_output, rating
        except:
            if maxtry <= 0:
                return llm_output, 0
            maxtry -= 1
            print(f"Not success! {maxtry} retries remaining...")
            time.sleep(random.uniform(1, 2))


# utils function for gpt evaluation
def get_llm_output(prompt, sys_prompt, max_tokens=128):
    if sys_prompt is None:
        sys_prompt = "You are an AI assistant for question answering."
    data = {"max_tokens": max_tokens, "model": "gpt-3.5-turbo-1106", "temperature": 1.0, "top_p": 1, "presence_penalty": 1, "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]}
    response = requests.post(API_URL, headers=headers, data=json.dumps(data).encode("utf-8"))
    result = response.content.decode("utf-8")
    dict_result = json.loads(result)
    llm_output = dict_result["choices"][0]["message"]["content"].strip()
    return llm_output


# utils function that converts gpt evaluation into rating
def llm_output_to_rating(llm_output):
    assert "Correct" in llm_output or "Incorrect" in llm_output
    if llm_output.startswith("Correct"):
        rating = 1
    elif llm_output.startswith("Incorrect"):
        rating = 0
    elif ("Correct" in llm_output) and ("Incorrect" not in llm_output):
        rating = 1
    elif "Incorrect" in llm_output:
        rating = 0
    return rating


# Factory into different aggregate
def tempcompass_aggregate_rating(results, args):
    yes_count = 0

    # results is a list of dict
    for answer_dict in results:
        if answer_dict["rating"] == 1:
            yes_count += 1

    accuracy = yes_count / len(results)

    return accuracy * 100
