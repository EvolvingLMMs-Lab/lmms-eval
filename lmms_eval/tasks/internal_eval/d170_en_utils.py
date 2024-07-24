import os
import requests
import time

import yaml
from pathlib import Path
import re
import json

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

from loguru import logger as eval_logger

with open(Path(__file__).parent / "d170_en.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
GPT_EVAL_MODEL_NAME = config["metadata"]["gpt_eval_model_name"]

# The EVALUATION_PROMPT_TEMPLATE_SIMPLE_V2 constant should be defined here
EVALUATION_PROMPT_TEMPLATE_SIMPLE_V2 = """You are an expert in judging the quality of a model response compared with given ground truth. The model response is in English while the ground truth can be in English or Chinese, or both. You should only judge the relevance of the model response to the ground truth based on meanings, not the language.
If the model response and ground truth are about grounding object coordinates, you may pay attention that the model responses are in format of [x_min, y_min, x_max, y_max]. You could judge the grounding quality by the IoU of the model response and the ground truth, or the distance between the center of the model response and the ground truth. If IoU is above 0.5 or the distance is below 0.3, you could give a score of 2. If IoU is below 0.2 or the distance is above 0.5, you could give a score of 0. If IoU is between 0.2 and 0.5 or the distance is between 0.2 and 0.5, you could give a score of 1.
Your response should be an integer score in [0, 1, 2], where 0 means the model response is completely irrelevant to the ground truth, and 2 means the model response completely matches the ground truth. You would have specific score criteria in the ground truth. You also need to explain your score in English.
Text: {prompt}
Ground Truth: {ground_truth}
You should response by following format:
Score:
Explanation:"""


def get_chat_response(prompt, model=GPT_EVAL_MODEL_NAME, max_tokens=512, patience=3, sleep_time=15):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    messages = [
        {"role": "user", "content": prompt},
    ]

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    while patience > 0:
        patience -= 1
        try:
            response = requests.post(
                API_URL,
                headers=headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            response_data = response.json()

            content = response_data["choices"][0]["message"]["content"].strip()
            if content != "":
                return content, response_data["model"]

        except Exception as e:
            eval_logger.info(f"Error in response: {response.json()['error']['message']}")
            if "Rate limit" in str(e):
                eval_logger.info("Sleeping due to rate limit...")
                time.sleep(sleep_time)
            eval_logger.info(f"Retrying...Patience left: {patience}")

    return "", ""


def doc_to_visual(doc):
    if doc["image"] is None:
        return []
    return [doc["image"].convert("RGB")]


def process_results(doc, results):
    # get pred and ground truth here
    pred = results[0]
    question = doc["question"]
    answer = doc["annotation"]
    gpt_query_prompt = EVALUATION_PROMPT_TEMPLATE_SIMPLE_V2.format(prompt=pred, ground_truth=answer)
    grade_sample_run_complete = False
    while not grade_sample_run_complete:
        try:
            response, model_name = get_chat_response(gpt_query_prompt)
            grade_sample_run_complete = True
        except Exception as e:
            eval_logger.info(f"Error in response: {e}")
            eval_logger.info(f"Retrying...")

    try:
        score = int(re.findall(r"Score:\s*(\d)", response)[0])
    except IndexError:
        score = 0  # Assign score 0 if the score wasn't parsed correctly

    return {
        "gpt_eval_info": {"question_id": doc["question_id"], "prediction": pred, "ground_truth": answer, "eval_model": model_name, "prompt": gpt_query_prompt, "response": response},
        "gpt_eval_avg_score": {
            "score": score,
        },
        "gpt_eval_score2_rate": {
            "score": score,
        },
    }


def d170_en_aggregate_info(results, args):
    path = generate_submission_file("dc170_en_eval_info.json", args)
    with open(path, "w") as f:
        json.dump(results, f)
    eval_logger.info(f"Results saved to {path}.")


def d170_en_aggregate_avg_score(results):
    total_score = 0
    for result in results:
        total_score += result["score"]
    avg_score = total_score / len(results)
    return avg_score


def d170_en_aggregate_score2_rate(results):
    score2_count = 0
    for result in results:
        if result["score"] == 2:
            score2_count += 1
    score2_rate = score2_count / len(results)
    return score2_rate
