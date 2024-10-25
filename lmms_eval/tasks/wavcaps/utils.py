import os
import re
import time
from pathlib import Path

import requests
import yaml
from loguru import logger as eval_logger


def wavcaps_doc_to_audio(doc):
    return [doc["context"]]


def wavcaps_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["instruction"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"


# functions for the clotho_asqa_v2 task, need to be tested later

with open(Path(__file__).parent / "wavcaps.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))


NUM_SECONDS_TO_SLEEP = 2
GPT_EVAL_MODEL_NAME = config["metadata"]["api_version"]
API_TYPE = os.getenv("API_TYPE", "azure")

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


# gpt-4
def get_eval(max_tokens: int, content: str):
    global headers

    time.sleep(NUM_SECONDS_TO_SLEEP)
    messages = [
        {"role": "user", "content": content},
    ]

    payload = {"model": GPT_EVAL_MODEL_NAME, "messages": messages, "temperature": 0, "max_tokens": max_tokens, "n": 1}

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        response_data = response.json()
        content = response_data["choices"][0]["message"]["content"].strip()

        if content != "":
            return content, payload["model"]
    except Exception as e:
        eval_logger.info(f"Attempt failed with error: {e}")
        return "", ""
    return "", ""


def wavcaps_process_results(doc, results):
    pred = results[0]
    ground_truth_str = doc["answer"]
    content = eval_prompt.format(model_response=pred, ground_truth=ground_truth_str, question=doc["instruction"])
    eval_answer, model_name = get_eval(max_tokens=1024, content=content)
    return {
        "gpt_eval": {"eval_answer": eval_answer, "model_name": model_name},
    }


def wavcaps_aggregate_results(results):
    score = 0
    for result in results:
        eval_answer = result["eval_answer"]

        try:
            match = re.search(r"Rating:\s*([0-5])\s*$", eval_answer)
            eval_score = match.group(1) if match else 0
            eval_score = float(eval_score)
        except Exception as e:
            eval_logger.error(f"Error parsing eval_score: {e}")
            eval_score = 0.0
        score += eval_score

    return score / len(results)
