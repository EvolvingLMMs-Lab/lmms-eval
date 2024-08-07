import json
import re
import os
import requests
import numpy as np
import time
import yaml
from pathlib import Path
from copy import deepcopy
from io import BytesIO
import base64

from loguru import logger as eval_logger

NUM_SECONDS_TO_SLEEP = 5


with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

GPT_EVAL_MODEL_NAME = config["metadata"]["judge_model"]
BASELINE_MODEL_NAME = config["metadata"]["baseline_model"]

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

system_prompt = """\
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.

Begin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.

When evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.

Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.

Then consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.

After providing your explanation, you must output only one of the following choices as your final verdict with a label:

1. Assistant A is significantly better: [[A>>B]]
2. Assistant A is slightly better: [[A>B]]
3. Tie, relatively the same: [[A=B]]
4. Assistant B is slightly better: [[B>A]]
5. Assistant B is significantly better: [[B>>A]]

Example output: "My final verdict is tie: [[A=B]]".\
"""

prompt_template = "<|User Prompt|>\n{question_1}\n\n<|The Start of Assistant A's Answer|>\n{answer_1}\n<|The End of Assistant A's Answer|>\n\n<|The Start of Assistant B's Answer|>\n{answer_2}\n<|The End of Assistant B's Answer|>"


def get_chat_response(base64_image, prompt, max_retries=5, wait_time=10):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": GPT_EVAL_MODEL_NAME,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64, {base64_image}"}},
                ],
            },
        ],
        "max_tokens": 1024,
        "temperature": 0.0,
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            response_data = response.json()
            print(response_data)
            return response_data["choices"][0]["message"]["content"], GPT_EVAL_MODEL_NAME
        except requests.exceptions.RequestException as e:
            print(f"Request failed on attempt {attempt+1}: {e}")
            if attempt == max_retries - 1:
                print(f"Failed to get response after {max_retries} attempts")
                return "", GPT_EVAL_MODEL_NAME
        except Exception as e:
            print(f"Error on attempt {attempt+1}: {e}")
            return "", GPT_EVAL_MODEL_NAME


def image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def get_score(judgement, pattern, pairwise=True):
    matches = pattern.findall(judgement)
    matches = [m for m in matches if m != ""]
    if len(set(matches)) == 0:
        return None, True
    elif len(set(matches)) == 1:
        if pairwise:
            return matches[0].strip("\n"), False
        return int(matches[0])
    else:
        return None, False


def wild_vision_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def wild_vision_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["instruction"].strip()
    if "pre_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["pre_prompt"] != "":
        question = f"{lmms_eval_specific_kwargs['pre_prompt']}{question}"
    if "post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["post_prompt"] != "":
        question = f"{question}{lmms_eval_specific_kwargs['post_prompt']}"
    return question


def wild_vision_doc_to_target(doc):
    return doc[BASELINE_MODEL_NAME]


def wild_vision_process_results(doc, results):
    pred = results[0]
    user_prompt = prompt_template.format(question_1=doc["instruction"], answer_1=doc[BASELINE_MODEL_NAME], answer_2=pred)
    base64_image = image_to_base64(doc["image"])
    resps, gpt_name = get_chat_response(base64_image, user_prompt)
    score, _ = get_score(resps, pattern=re.compile("\[\[([AB<>=]+)\]\]"))

    if score is None:
        score = resps

    if "A>B" in score:
        final_score = -1
        judgement = "Worse"  # Baseline better
    elif "A>>B" in score:
        final_score = -2
        judgement = "Worse++"
    elif "A=B" in score:
        final_score = 0
        judgement = "Tie"
    elif "B>A" in score:
        final_score = 1
        judgement = "Better"
    elif "B>>A" in score:
        final_score = 2
        judgement = "Better++"
    else:
        final_score = 0
        judgement = "Unclear"

    return {"gpt_eval_score": {"question": doc["instruction"], "score": final_score, "gpt_resps": resps, "ans_1": doc[BASELINE_MODEL_NAME], "ans_2": pred, "filtered_resps": score, "judgement": judgement}}


def wild_vision_aggregation(results):
    score = 0
    for res in results:
        score += res["score"]

    return score / len(results)
