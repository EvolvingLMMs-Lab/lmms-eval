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

eval_logger = logging.getLogger("lmms-eval")


# Pass in video path here
# Can only work correctly with video llm
def vitatecs_doc_to_visual(doc):
    video_path = os.path.join(cache_dir, doc["src_dataset"], doc["video_name"])
    if os.path.exists(video_path):
        video_path = video_path
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


# This is the place where you format your question
def vitatecs_doc_to_text(doc, model_specific_prompt_kwargs=None):
    if model_specific_prompt_kwargs is None:
        model_specific_prompt_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in model_specific_prompt_kwargs:
        pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
    if "post_prompt" in model_specific_prompt_kwargs:
        post_prompt = model_specific_prompt_kwargs["post_prompt"]

    question, _, _ = format_question_and_answer(doc)
    return f"{pre_prompt}{question}{post_prompt}"


def process_option_for_question(sent):
    if not sent.endswith("."):
        sent += "."
    return sent.capitalize()


def process_option_for_matching(sent):
    if sent.endswith("."):
        sent = sent[:-1]
    return sent.lower()


def format_question_and_answer(doc):
    seed = sum(ord(c) for c in doc['caption'] + doc['counterfactual']) % 100
    random.seed(seed)
    if random.random() > 0.5:
        option_a = process_option_for_question(doc['caption'])
        option_b = process_option_for_question(doc['counterfactual'])
        answer = "(A) " + option_a
    else:
        option_a = process_option_for_question(doc['counterfactual'])
        option_b = process_option_for_question(doc['caption'])
        answer = "(B) " + option_b
    options = [process_option_for_matching(doc['caption']), process_option_for_matching(doc['counterfactual'])]

    question = f"Which of the following best describes the content of the video: \n(A) {option_a} \n(B) {option_b}"
    return question, answer, options


def vitatecs_doc_to_answer(doc):
    _, answer, _ = format_question_and_answer(doc)
    return answer


# Process result
def vitatecs_process_results(doc, result):
    pred = result[0]
    rating = 0
    match_success = True
    chatgpt_response = None
    question, answer, options = format_question_and_answer(doc)

    # Some hand-crafted matching rules
    if options[0] in pred.lower() and options[1] not in pred.lower():
        rating = 1
    elif options[1] in pred.lower() and options[0] not in pred.lower():
        rating = 0
    elif pred in ["A", "B"]:
        rating = 1 if pred == answer[1] else 0
    elif any(pred.startswith(prefix) for prefix in ["A.", "B."]):
        rating = 1 if pred.split(".")[0] == answer[1] else 0
    elif any(pred.startswith(prefix) for prefix in ["A)", "B)"]):
        rating = 1 if pred.split(")")[0] == answer[1] else 0
    elif any(pred.startswith(prefix) for prefix in ["(A)", "(B)"]):
        rating = 1 if pred.split(")")[1] == answer[1] else 0
    else:
        # Fail to match answer in the video-llm response. Use ChatGPT to evaluate.
        match_success = False

        base_prompt = """You will receive a caption matching question, the ground-truth answer and the prediction from a question answering (QA) model. Your task is to determine whether QA model prediction is correct, based on the question and ground-truth answer. If the prediction is correct, respond "Correct". If the prediction is incorrect, respond "Incorrect". """
        prompt = f"""{base_prompt}\n\nCaption Matching Question: {question}\n\nGround-Truth Answer: {answer}\n\nModel Prediction: {pred}"""
        chatgpt_response, rating = get_eval_result(prompt)

    if not match_success:
        return {
            "accuracy": {
                "src_dataset": doc["src_dataset"],
                "video_id": doc["video_name"],
                "question": question,
                "gt-answer": answer,
                "video-llm-prediction": pred,
                "match_success": match_success,
                "rating": rating,
                # "chatgpt_prompt": prompt, 
                "chatgpt_response": chatgpt_response,
                "aspect": doc["aspect"],
            },
        }
    else:
        return {
            "accuracy": {
                "src_dataset": doc["src_dataset"],
                "video_id": doc["video_name"],
                "question": question,
                "gt-answer": answer,
                "video-llm-prediction": pred,
                "match_success": match_success,
                "rating": rating,
                "aspect": doc["aspect"],
            },
        }


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
def vitatecs_aggregate_rating(results, args):
    yes_count = 0

    # results is a list of dict
    for answer_dict in results:
        if answer_dict["rating"] == 1:
            yes_count += 1

    accuracy = yes_count / len(results)

    return accuracy * 100
