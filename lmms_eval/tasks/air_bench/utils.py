import datetime
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import requests
import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file


def air_bench_doc_to_audio(doc):
    return [doc["audio"]]


def air_bench_doc_to_text_chat(doc, lmms_eval_specific_kwargs):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"


with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

# specify api type and key in .env
GPT_EVAL_MODEL_NAME = config["metadata"]["gpt_eval_model_name"]
API_TYPE = os.getenv("API_TYPE", "azure")

if API_TYPE == "openai":
    API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
elif API_TYPE == "azure":
    API_KEY = os.getenv("AZURE_API_KEY", None)
    deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME", None)
    resource_name = os.getenv("AZURE_RESOURCE_NAME", None)
    api_version = os.getenv("AZURE_API_VERSION", None)
    API_URL = f"https://{resource_name}.openai.azure.com/openai/deployments/{deployment_name}/chat/completions?api-version={api_version}"
    headers = {
        "api-key": API_KEY,
        "Content-Type": "application/json",
    }

# prompt taken from the AIR-Bench repo
eval_prompt = (
    "You are a helpful and precise assistant for checking the quality of the answer.\n"
    "[Detailed Audio Description]\nXAudioX\n[Question]\nXQuestionX\n"
    "[The Start of Assistant 1s Answer]\nXAssistant1X\n[The End of Assistant 1s Answer]\n"
    "[The Start of Assistant 2s Answer]\nXAssistant2X\n[The End of Assistant 2s Answer]\n[System]\n"
    "We would like to request your feedback on the performance of two AI assistants in response to the user question "
    "and audio description displayed above. AI assistants are provided with detailed audio descriptions and questions.\n"
    "Please rate the helpfulness, relevance, accuracy, and comprehensiveness of their responses. "
    "Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance. "
    "Please output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. "
    "The two scores are separated by a space. Please only output the 2 required number and no text."
)

retries = 3
NUM_SECONDS_TO_SLEEP = 5


def get_eval(max_tokens: int, content: str, retries: int = retries):
    global headers

    messages = [
        {"role": "user", "content": content},
    ]

    payload = {
        "model": GPT_EVAL_MODEL_NAME,
        "messages": messages,
        "temperature": 0,
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
                return "", ""
    return "", ""


def air_bench_process_results_chat(doc, result):
    path = doc["path"]
    question = doc["question"]
    answer_gt = doc["answer_gt"]
    task_name = doc["task_name"]
    dataset_name = doc["dataset_name"]
    response = result[0]

    if response == None:
        exit(1)

    if doc["meta_info"] == None:
        print("lack meta info")
        exit(1)
    else:
        meta_info = doc["meta_info"]

    # Get the evaluation score 2 times: one with ourmodel as assistant 1 and the other with our model as assistant 2 to prevent position bias
    content = eval_prompt.replace("XAudioX", meta_info).replace("XQuestionX", question).replace("XAssistant1X", answer_gt).replace("XAssistant2X", response)
    eval_answer, model_name = get_eval(max_tokens=1024, content=content)
    content = eval_prompt.replace("XAudioX", meta_info).replace("XQuestionX", question).replace("XAssistant1X", response).replace("XAssistant2X", answer_gt)
    eval_answer2, model_name2 = get_eval(max_tokens=1024, content=content)

    return {
        "gpt_eval": {"eval_answer": [eval_answer, eval_answer2], "model_name": model_name},
    }


def air_bench_aggregate_results_chat(results):
    score = 0
    for result in results:
        eval_answer = result["eval_answer"]
        eval_answer = [eval_answer[i].strip().replace(".", "") for i in range(len(eval_answer))]
        pattern = r"\b(?:[1-9]|10)\b"

        # Find all matches
        try:
            matches1 = re.findall(pattern, eval_answer[0])
            matches2 = re.findall(pattern, eval_answer[1])
            # Get the first two occurrences
            eval_score = float(matches1[1]) + float(matches2[0])
            score += eval_score
        except Exception as e:
            eval_logger.error(f"Error parsing eval_score: {e}")
            continue

    return score / (2 * len(results))


# Functions for Foundation tasks


def air_bench_doc_to_text_foundation(doc, lmms_eval_specific_kwargs):
    question = doc["question"]
    answers = [doc["choice_a"], doc["choice_b"], doc.get("choice_c", None), doc.get("choice_d", None)]
    question = f"{question}\nA. {answers[0]}\nB. {answers[1]}\nC. {answers[2]}\nD. {answers[3]}\n"
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"


def air_bench_doc_to_target_foundation(doc):
    return get_gt(doc)


def air_bench_doc_to_choice_foundation(doc):
    choices = []
    for option in ["choice_a", "choice_b", "choice_c", "choice_d"]:
        if doc.get(option) and doc[option].strip():
            choices.append(option[-1].upper())
    return choices


def air_bench_process_results_foundation(doc, result):
    response = result[0].strip()
    all_choices = [choice[-1].upper() for choice in ["choice_a", "choice_b", "choice_c", "choice_d"] if doc.get(choice)]
    pred = parse_multi_choice_response(response, all_choices)  # AdaptfromMMMU
    gt_ans = get_gt(doc)
    score = 1.0 if pred == gt_ans else 0.0
    submission_dict = {}
    submission_dict = {doc.get("uniq_id", "unknown"): pred}
    return {"accuracy": {"score": score, "task": doc["task_name"]}, "submission": submission_dict}


def air_bench_aggregate_results_for_submission(results, args):
    path = generate_submission_file("air_bench_test_submission.json", args)
    with open(path, "w") as f:
        json.dump(results, f)
    eval_logger.info(f"Results saved to {path}.")


def air_bench_aggregate_results_foundation(results):
    score = 0
    categorical_correct = {}
    categorical_total = {}
    for result in results:
        score += result["score"]
        if result["task"] not in categorical_correct.keys():
            categorical_correct[result["task"]] = 0
            categorical_total[result["task"]] = 0
        categorical_correct[result["task"]] += result["score"]
        categorical_total[result["task"]] += 1

    return {"overall_accuracy": score / len(results), "categorical_accuracy": {task: categorical_correct[task] / categorical_total[task] for task in categorical_correct.keys()}}


def parse_multi_choice_response(response, all_choices):
    """
    Parse the prediction from the generated response.
    Return the predicted choice letter e.g., A, B, C, D.
    """
    # Clean response of unwanted characters
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # Add space to avoid partial match

    candidates = []
    # Look for choices with parentheses, e.g., (A)
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)

    # Look for simple choices, e.g., A, B, C
    if len(candidates) == 0:
        for choice in all_choices:
            if f" {choice} " in response:
                candidates.append(choice)

    # Look for choices with periods, e.g., A., B., C.
    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice}." in response:
                candidates.append(choice)

    # If no candidates, randomly choose one
    if len(candidates) == 0:
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        # If more than one candidate, choose the last one found
        start_indexes = [response.rfind(f" {can} ") for can in candidates]
        pred_index = candidates[np.argmax(start_indexes)]
    else:
        # If only one candidate, use it
        pred_index = candidates[0]

    return pred_index


def get_gt(doc):
    if doc["answer_gt"] == doc["choice_a"]:
        return "A"
    elif doc["answer_gt"] == doc["choice_b"]:
        return "B"
    elif doc["answer_gt"] == doc.get("choice_c", None):
        return "C"
    elif doc["answer_gt"] == doc.get("choice_d", None):
        return "D"
