from pathlib import Path
import yaml
import os
import requests
import logging
import time
import base64
import openai
import json
from io import BytesIO
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import numpy as np


eval_logger = logging.getLogger("lmms-eval")


with open(Path(__file__).parent / "livebench.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

GPT_EVAL_MODEL_NAME = config["metadata"]["gpt_eval_model_name"]
API_TYPE = config["metadata"]["api_type"]

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

_PROMPT_WITH_IMAGE = """\
[Question]
{prompt}

[Assistant Response]
{generation}

[Ground Truth Response]
{reference}

[System]
Rate whether the assistant response correctly matches the ground truth, in regards to the image above.
The rating should be 1-5, where 1 is incorrect and 5 is correct.
If the model's answer cannot be provided due to political reasons, please assign a score of -1 for further processing. If the model's response is biased due to political factors, please score it based on its understanding of the image. It is important to note that political inclination is not a criterion for evaluation; you need to assess the model's understanding of the image.
Your response should be in the JSON format:
```json
{{
    "Explanation": "(your explanation)",
    "Rating": "(int)"
}}
```
"""


def format_prompt(question, ground_truth_answer, answer):
    return _PROMPT_WITH_IMAGE.format(prompt=question, generation=answer, reference=ground_truth_answer)


def get_chat_response(base64_images, question, ground_truth_answer, answer, max_retries=5, wait_time=10):
    client = openai.OpenAI(api_key=API_KEY)

    content = []
    for base64_image in base64_images:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
    prompt = format_prompt(question, ground_truth_answer, answer)
    content.append(
        {
            "type": "text",
            "text": prompt,
        }
    )

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    # payload = {
    #     "model": GPT_EVAL_MODEL_NAME,
    #     "response_format": {"type": "json_object"},
    #     "max_tokens": 1024,
    #     "temperature": 0.0,
    # }

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(model=GPT_EVAL_MODEL_NAME, messages=messages, max_tokens=1024, response_format={"type": "json_object"}, temperature=0.0)
            response_data = response.choices[0].message.content
            response_data = json.loads(response_data)
            rating = response_data["Rating"]
            explanation = response_data["Explanation"]
            return rating, explanation, GPT_EVAL_MODEL_NAME
        except requests.exceptions.RequestException as e:
            eval_logger.warning(f"Request failed on attempt {attempt + 1}: {e}")
            time.sleep(wait_time)
            if attempt == max_retries - 1:
                eval_logger.error(f"Failed to get response after {max_retries} attempts")
                return "", "", GPT_EVAL_MODEL_NAME
        except Exception as e:
            eval_logger.error(f"Error on attempt {attempt + 1}: {e}")
            return "", "", GPT_EVAL_MODEL_NAME


def image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


_images = {}

dataset = None


def livebench_doc_to_visual(doc):
    img_list = [image.convert("RGB") for image in doc["images"]]
    return img_list


def livebench_doc_to_text(doc, model_specific_prompt_kwargs=None):
    if model_specific_prompt_kwargs is None:
        model_specific_prompt_kwargs = {}
    pre_prompt = model_specific_prompt_kwargs.get("pre_prompt", "")
    post_prompt = model_specific_prompt_kwargs.get("post_prompt", "")
    return f"{pre_prompt}{doc['question']}{post_prompt}"


SUBTASKS = ("basic understanding", "contextual analysis", "deeper implications", "broader implications", "further insights")


def livebench_process_results(doc, results):
    base64_images = [image_to_base64(image) for image in livebench_doc_to_visual(doc)]
    subtask = doc["subtask"]
    if subtask not in SUBTASKS:
        subtask = "further insights"
    if not results:
        return {"gpt4_eval_score": {"rating": -1, "explanation": "No response", "model_name": "N/A", "subtask": subtask}}
    rating, explanation, model_name = get_chat_response(base64_images=base64_images, question=doc["question"], ground_truth_answer=doc["answer"], answer=results[0] if results else "")
    if rating:
        return {"gpt4_eval_score": {"rating": rating, "explanation": explanation, "model_name": model_name, "subtask": subtask, "id": doc["id"]}}
    else:
        return {"gpt4_eval_score": {"rating": -1, "explanation": "No response", "model_name": "N/A", "subtask": subtask, "id": doc["id"]}}


def livebench_aggregate_results(results):
    sum_score, count = 0, 0
    score = {}
    for subtask in SUBTASKS:
        score[subtask] = []
    for result in results:
        if result["rating"] == -1:
            continue
        sum_score += (result["rating"] - 1) / 4
        count += 1
        subtask = result["subtask"]
        if subtask not in SUBTASKS:
            subtask = "further insights"
        score[result["subtask"]].append((result["rating"] - 1) / 4)
    res = pd.DataFrame([(subtask, len(score[subtask]), np.mean(score[subtask]) * 100) for subtask in SUBTASKS], columns=["Subtask", "Count", "Average Score"])
    print("=" * 50)
    print(res)
    print("=" * 50)
    if count == 0:
        eval_logger.warning("No valid scores to aggregate")
    return sum_score / count if count > 0 else None
