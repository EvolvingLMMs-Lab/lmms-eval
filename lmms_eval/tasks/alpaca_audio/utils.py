import datetime
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import requests
import yaml
from loguru import logger as eval_logger

import lmms_eval.tasks._task_utils.file_utils as file_utils
from lmms_eval.filters.extraction import ExtendedRegexFilter


def doc_to_audio(doc):
    return [doc["context"]]


def doc_to_text(doc, lmms_eval_specific_kwargs):
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{post_prompt}"


with open(Path(__file__).parent / "alpaca_audio.yaml", "r") as f:
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

eval_prompt = """
            [Question]
            {question}

            [Reference Answer]
            {ground_truth}

            [Model Answer]
            {model_response}

            [Task]
            Rate the model's answer based on its alignment with the reference answer, focusing on accuracy and relevance to the reference provided. Please be critical on the details.
            Criteria: Assess if the model's response mirrors the reference in terms of content, accuracy, and relevance.
            Score0: The answer is completely misaligned, providing incorrect or irrelevant information compared to the reference.
            Score1: The answer shows minimal alignment, often misunderstanding or providing irrelevant details unrelated to the reference.
            Score2: The answer recognizes the topic but diverges significantly from the reference in accuracy or relevance.
            Score3: The answer aligns with the reference generally but lacks detail or precise accuracy in some aspects.
            Score4: The answer is mostly accurate and relevant, closely following the reference but could be clearer or more detailed.
            Score5: The answer is highly accurate, detailed, and matches the reference answer perfectly, capturing its essence and detail.

            Your response should be formatted as follows:
            Explanation: (Provide a concise explanation of your rating, comparing the reference answer with the model's response. "The reference answer is [XXX], while the model's answer is [YYY]. I think ...")
            Rating: (int)"""


def get_eval(max_tokens: int, content: str):
    global headers

    messages = [
        {"role": "user", "content": content},
    ]

    payload = {"model": GPT_EVAL_MODEL_NAME, "messages": messages, "temperature": 0.7, "max_tokens": max_tokens, "top_p": 0.95, "frequency_penalty": 0, "presence_penalty": 0, "stop": None}

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        response_data = response.json()

        content = response_data["choices"][0]["message"]["content"].strip()
        if content != "":
            return content, response_data["model"]
    except Exception as e:
        eval_logger.info(f"Attempt failed with error: {e}")
        return "", ""
    return "", ""


def alpaca_audio_process_results(doc, result):
    pred = result[0]
    ground_truth_str = doc["answer"]
    content = eval_prompt.format(model_response=pred, ground_truth=ground_truth_str, question=doc["speech_instruction"])
    eval_answer, model_name = get_eval(max_tokens=1024, content=content)
    return {
        "gpt_eval": {"eval_answer": eval_answer, "model_name": model_name},
    }


def alpaca_audio_aggregate_results(results):
    score = 0
    for result in results:
        try:
            eval_answer = result["eval_answer"]
            eval_score = re.search(r"([0-5])", eval_answer).group(1)
            eval_score = float(eval_score)
        except Exception as e:
            eval_logger.error(f"Error parsing eval_score: {e}")
            eval_score = 0.0
        score += eval_score

    return score / len(results) * 20
