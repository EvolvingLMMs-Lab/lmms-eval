import json
import os
import re
import time
from pathlib import Path

import requests
import yaml
from loguru import logger as eval_logger

from lmms_eval.llm_judge import ServerConfig, get_server
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

with open(Path(__file__).parent / "d170_en.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

API_TYPE = os.getenv("API_TYPE", "openai")
MODEL_VERSION = os.getenv("MODEL_VERSION", "gpt-4o-2024-08-06")

# Initialize the judge server
server_config = ServerConfig(
    model_name=MODEL_VERSION,
)
server = get_server(server_name=API_TYPE, config=server_config)

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

    # Define custom prompt for D170 EN evaluation
    custom_prompt = """You are an expert in judging the quality of a model response compared with given ground truth.

If the model response and ground truth are about grounding object coordinates, you may pay attention that the model responses are in format of [x_min, y_min, x_max, y_max]. You could judge the grounding quality by the IoU of the model response and the ground truth, or the distance between the center of the model response and the ground truth:
- If IoU is above 0.5 or the distance is below 0.3, score 1 (correct)
- If IoU is below 0.2 or the distance is above 0.5, score 0 (incorrect)
- For other cases, score 0

For non-grounding questions:
- Score 1 if the prediction matches the answer semantically, it can be in different format
- Score 0 for incorrect, partially correct, or answers with extra incorrect information

Return only "1" or "0" with no additional text or formatting."""

    try:
        # Use the llm_judge API for binary evaluation
        result = server.evaluate_binary(question=question, answer=str(answer), prediction=pred, output_format="0/1", custom_prompt=custom_prompt)

        # Parse the result
        if result["success"]:
            judge_response = result["result"]
            judge_score = judge_response.strip()
            score = 1 if judge_score == "1" else 0
        else:
            eval_logger.error(f"Judge evaluation failed: {result.get('raw_response', 'Unknown error')}")
            score = 0
    except Exception as e:
        eval_logger.error(f"Error getting judge response: {e}")
        score = 0

    return {"llm_as_judge_eval": score}


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
