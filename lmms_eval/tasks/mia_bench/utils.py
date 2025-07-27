import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests
import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file


def mia_bench_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def mia_bench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question_text = doc["instruction"]

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") if lmms_eval_specific_kwargs else ""

    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "") if lmms_eval_specific_kwargs else ""
    formatted_question = f"{pre_prompt}{question_text}{post_prompt}"

    return formatted_question


# ============================
# Result Processing Functions
# ============================

with open(Path(__file__).parent / "mia_bench.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

NUM_SECONDS_TO_SLEEP = 10
API_TYPE = os.getenv("API_TYPE", "openai")
GPT_EVAL_MODEL_NAME = os.getenv("MODEL_VERSION", "gpt-4o-2024-11-20")

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
    headers = {"api-key": API_KEY, "Content-Type": "application/json", "api-version": "2023-07-01-preview"}


def get_eval(content: str, max_tokens: int, retries: int = 5):
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


def generate_prompt(d, response):
    instruction = d["instruction"]
    weight = d["component_weight"] * 1
    d["num_of_component"] = len(d["components"])
    for i in range(len(weight)):
        weight[i] = str(weight[i])
    if d["num_of_component"] == 1:
        components = """The first component is:' """ + d["components"][0] + "'"
        score = """The first component is worth """ + weight[0] + " scores."
    elif d["num_of_component"] == 2:
        components = """The first component is:' """ + d["components"][0] + """', and the second component is:' """ + d["components"][1] + "'"
        score = """The first and second component is each worth """ + weight[0] + " and " + weight[1] + " scores."
    elif d["num_of_component"] == 3:
        components = """The first component is:' """ + d["components"][0] + """', and the second component is:' """ + d["components"][1] + """', and the third component is:' """ + d["components"][2] + "'"
        score = """The first second, and third component is each worth """ + weight[0] + ", " + weight[1] + " and " + weight[2] + " scores."
    elif d["num_of_component"] == 4:
        components = (
            """The first component is:' """
            + d["components"][0]
            + """', and the second component is:' """
            + d["components"][1]
            + """', and the third component is:' """
            + d["components"][2]
            + """', and the fourth component is:' """
            + d["components"][3]
            + "'"
        )
        score = """The first second, third, and fourth component is each worth """ + weight[0] + ", " + weight[1] + ", " + weight[2] + " and " + weight[3] + " scores."
    elif d["num_of_component"] == 5:
        components = (
            """The first component is:' """
            + d["components"][0]
            + """', and the second component is:' """
            + d["components"][1]
            + """', and the third component is:' """
            + d["components"][2]
            + """', and the fourth component is:' """
            + d["components"][3]
            + """', and the fifth component is:' """
            + d["components"][4]
            + "'"
        )
        score = """The first second, third, fourth and fifth component is each worth """ + weight[0] + ", " + weight[1] + ", " + weight[2] + ", " + weight[3] + " and " + weight[4] + " scores."
    return (
        """Here is an instruction for a multimodal LLM: ' """
        + instruction
        + """ You need to grade if the response from the model follows each component of the instruction. """
        + components
        + """ The response is:' """
        + response
        + """' You need to score the response and be strict. The total score ranges from 0 to 10, depending on if the response follows the instruction. """
        + score
        + " List scores of each component, and the total score in one sentence in this format: score of component 1: x/2, score of component 2: y/8, total score: z/10. Then explain your reasons."
    )


def process_rawscore(component_type, raw_score):
    first_sentence = raw_score.split(""".""")[0].split(""",""")
    score_dict = {}
    for i in range(len(first_sentence) - 1):
        score_ = first_sentence[i].split(""":""")[1][1:].split("""/""")
        score = int(score_[0]) / int(score_[1])
        score_dict[component_type[i]] = score

    # Ensure that the loop has executed at least once
    if len(first_sentence) > 1:
        total_score_ = first_sentence[-1].split(""":""")[1][1:].split("""/""")
        total_score = int(total_score_[0]) / int(total_score_[1])
        score_dict["total_score"] = total_score
    else:
        score_dict["total_score"] = 0  # or handle this case as needed

    return score_dict


def mia_bench_process_results(doc, results):
    response = results[0].strip()
    components = doc["components"]
    eval_prompt = generate_prompt(doc, response)
    eval_score, _ = get_eval(eval_prompt, 1024)
    score_dict = process_rawscore(components, eval_score)
    return {"gpt_eval_score": score_dict}


# ============================
# Aggregation Functions
# ============================


def mia_bench_aggregate_results(results):
    total_score = 0
    for result in results:
        # Overall accuracy
        total_score += result["total_score"]
    return total_score / len(results)
