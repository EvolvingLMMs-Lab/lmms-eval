import ast
import datetime
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import openai
import requests
import yaml
from loguru import logger as eval_logger
from openai import OpenAI
from tqdm import tqdm

import lmms_eval.tasks._task_utils.file_utils as file_utils

dir_name = os.path.dirname(os.path.abspath(__file__))

one_score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
one_score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")

with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

API_ERROR_OUTPUT = "$ERROR$"

API_MAX_RETRY = 6

NUM_SECONDS_TO_SLEEP = 15

GPT_EVAL_MODEL_NAME = config["metadata"]["gpt_eval_model_name"]

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
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
else:
    API_URL = "YOUR_API_URL"
    API_KEY = "YOUR_API_KEY"


def egothink_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


# format the question
def egothink_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"].strip()
    if "pre_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["pre_prompt"] != "":
        question = f"{lmms_eval_specific_kwargs['pre_prompt']}{question}"
    if "post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["post_prompt"] != "":
        question = f"{question}{lmms_eval_specific_kwargs['post_prompt']}"
    return question


# format answer
def egothink_doc_to_answer(doc):
    return doc["answer"]


# Process result for evaluation in generic task
def chat_compeletion_openai(model, messages, temperature, max_tokens):
    # headers = {
    #     "Authorization": f"Bearer {API_KEY}",
    #     "Content-Type": "application/json",
    # }
    # headers = {
    #     "Authorization": f"Bearer {API_KEY}",
    #     "Content-Type": "application/json",
    # }
    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY,
    }
    output = API_ERROR_OUTPUT
    payload = {
        # "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt in range(API_MAX_RETRY):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()  # Raises HTTPError for bad responses
            try:
                response_data = response.json()  # Attempt to parse JSON
            except requests.exceptions.JSONDecodeError:
                eval_logger.error(f"JSON decode error on attempt {attempt + 1}. Response text: {response.text}")
                continue  # Skip to next retry
            content = response_data["choices"][0]["message"]["content"].strip()
            if content != "":
                return content, response_data["model"]
        # Handle HTTP errors separately
        except requests.exceptions.HTTPError as e:
            eval_logger.error(f"HTTP error on attempt {attempt + 1}: {e}")
        # Handle other requests-related errors
        except requests.exceptions.RequestException as e:
            eval_logger.error(f"Request exception on attempt {attempt + 1}: {e}")
        except Exception as e:
            eval_logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")

        # Handle other unexpected errors
        if attempt < API_MAX_RETRY - 1:
            time.sleep(NUM_SECONDS_TO_SLEEP)
        else:  # If this was the last attempt, log and return empty
            eval_logger.error(f"All {retries} attempts failed. Last error message: {e}")
            return "", ""

    return "", ""


def judge_single(question, answer, ref_answer):
    model = GPT_EVAL_MODEL_NAME

    rating = -1

    conv = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": f"[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider correctness and helpfulness. You will be given a reference answer and the assistant's answer. Begin your evaluation by comparing the assistant's answer with the reference answer. Identify and correct any mistakes. The assistant has access to an image alongwith questions but you will not be given images. Therefore, please consider only how the answer is close to the reference answer. If the assistant's answer is not exactly same as or similar to the answer, then he must be wrong.  Be as objective as possible. Discourage uninformative answers. Also, equally treat short and long answers and focus on the correctness of answers.  After providing your explanation, you must rate the response with either 0, 0.5 or 1 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[0.5]]\".\n\n[Question]\n{question}\n\n[The Start of Reference Answer]\n{ref_answer}\n[The End of Reference Answer]\n\n[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant's Answer]",
        },
    ]

    judgment, eval_model = chat_compeletion_openai(model, conv, temperature=0, max_tokens=2048)
    for _ in range(3):
        match = re.search(one_score_pattern, judgment)
        if not match:
            match = re.search(one_score_pattern_backup, judgment)

        if match:
            rating = ast.literal_eval(match.groups()[0])
            break
        else:
            rating = -1
    return rating, judgment, eval_model


def egothink_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case mme score), value: metric value
    """
    pred = results[0]
    question = doc["question"]
    ref_ans = doc["answer"].lower().strip().replace(".", "")
    score, judge, eval_model = judge_single(question, pred, ref_ans)
    return {"gpt_eval_score": {"question_id": doc["id"], "score": score, "judge": judge, "eval_model": eval_model}}


def egothink_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    total_score = 0
    for result in results:
        total_score += result["score"]
    return total_score / len(results)
