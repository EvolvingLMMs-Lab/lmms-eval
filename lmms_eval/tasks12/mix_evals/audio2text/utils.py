import datetime
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
import yaml
from loguru import logger as eval_logger

import lmms_eval.tasks._task_utils.file_utils as file_utils
from lmms_eval.filters.extraction import ExtendedRegexFilter

with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

NUM_SECONDS_TO_SLEEP = 5
GPT_EVAL_MODEL_NAME = os.getenv("MODEL_VERSION", "gpt-4o-2024-11-20")
API_TYPE = os.getenv("API_TYPE", "openai")

if API_TYPE == "openai":
    API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
elif API_TYPE == "azure":
    API_URL = os.getenv("AZURE_ENDPOINT", "https://api.cognitive.microsoft.com/sts/v1.0/issueToken")
    API_KEY = os.getenv("AZURE_API_KEY", "YOUR_API_KEY")
    headers = {
        "api-key": API_KEY,
        "Content-Type": "application/json",
    }

eval_prompt = """You are an AI assistant who will help me to evaluate the quality of a model response to a few candidate ground truth answers.

Some criterion
- Response that perfectly reflect the meaning of the ground truth: 1 point
- Response that reflect none of the key points in the ground truth: 0 point
- Some part in the response are correct but some parts in the ground truth are not mentioned in the response: 0.5 point
- Some part in the response are correct but other parts in the response are not mentioned in the ground truth: 0.5 point

Here're some examples about the scoring criterion and format:
model response: Steam Cleaning Services
ground truth: ["steam clean", "steam clean", "cleaning", "car", "steam clean"],
Point: 1

model response: A cowboy action shooter.
ground truth: ["man"]
Point: 1

model response: I'm sorry, but I can't assist with that request.
ground truth: ["quality"]
Point: 0

Let's begin this task:
model response: {model_response}
ground truth: {ground_truth}
Point:"""


def get_eval(model_response: str, ground_truth: str, max_tokens: int, retries: int = 5):
    global headers
    content = eval_prompt.format(model_response=model_response, ground_truth=ground_truth)

    messages = [
        {"role": "user", "content": content},
    ]

    payload = {
        "model": GPT_EVAL_MODEL_NAME,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }

    for attempt in range(retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            response_data = response.json()

            content = response_data["choices"][0]["message"]["content"].strip()
            if content != "":
                return content, response_data["model"]
            break  # If successful, break out of the loop

        except Exception as e:
            eval_logger.info(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < retries:  # If we have retries left, sleep and then continue to next attempt
                time.sleep(NUM_SECONDS_TO_SLEEP)
            else:  # If this was the last attempt, log and return empty
                eval_logger.error(f"All {retries} attempts failed. Last error message: {e}")
                return "0", ""
    return "", ""


def mix_evals_audio2text_doc_to_audio(doc):
    return [doc["audio"]]


def mix_evals_audio2text_doc_to_target(doc):
    return doc["reference_answer"][0]


# This is the place where you format your question
def mix_evals_audio2text_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    if "post_prompt" in lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs["post_prompt"]

    user_prompt = doc["query"]

    if pre_prompt:
        user_prompt = f"{pre_prompt}\n{user_prompt}"

    if post_prompt:
        user_prompt = f"{user_prompt}\n{post_prompt}"
    return user_prompt


def mix_evals_audio2text_process_results_freeform(doc, result):
    pred = result[0]
    ground_truth_str = doc["reference_answer"][0]
    content = eval_prompt.format(model_response=pred, ground_truth=ground_truth_str)
    eval_answer, model_name = get_eval(model_response=pred, ground_truth=ground_truth_str, max_tokens=1024)
    return {
        "gpt_eval": {"pred": pred, "id": doc["id"], "target": ground_truth_str, "eval_answer": eval_answer, "gpt_prompt": content},
    }


def mix_evals_audio2text_gpt_eval(results, args):
    score = 0
    for result in results:
        eval_answer = result["eval_answer"]
        eval_score = re.search(r"([0-9.]+)", eval_answer).group(1)
        try:
            eval_score = float(eval_score)
        except Exception as e:
            eval_logger.error(f"Error parsing eval_score: {e}")
            eval_score = 0.0
        score += eval_score

    return score / len(results)
