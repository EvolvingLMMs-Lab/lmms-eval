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
from tqdm import tqdm
import pandas as pd
import numpy as np


eval_logger = logging.getLogger("lmms-eval")


with open(Path(__file__).parent / "live_bench.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

API_TYPE = config["metadata"]["api_type"]
EVAL_WITH_MINI = config["metadata"]["eval_with_mini"]

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

The rating should be 0-10, where 0 is incorrect and 10 is correct.

Below is the specific criteria for rating:

{criteria}

Total score is out of 10. If the model's answer cannot be provided due to political reasons, please assign a score of 0 for further processing. If the model's response is biased due to political factors, please score it based on its understanding of the image, but reduce the objectivity score accordingly.

Your response should be in the JSON format:
```json
{{
    "Explanation": "(your explanation)",
    "Rating": "(int)"
}}
```
"""


def format_prompt(question, ground_truth_answer, answer, criteria):
    return _PROMPT_WITH_IMAGE.format(prompt=question, generation=answer, reference=ground_truth_answer, criteria=criteria)


def get_chat_response(gpt_model_name, base64_images, question, ground_truth_answer, answer, criteria, max_retries=5, wait_time=10):
    client = openai.OpenAI(api_key=API_KEY)

    content = []
    for base64_image in base64_images:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
    prompt = format_prompt(question, ground_truth_answer, answer, criteria)
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
            response = client.chat.completions.create(model=gpt_model_name, messages=messages, max_tokens=1024, response_format={"type": "json_object"}, temperature=0.0)
            response_data = response.choices[0].message.content
            # print(response_data)
            response_data = json.loads(response_data)
            rating = response_data["Rating"]
            explanation = response_data["Explanation"]
            return rating, explanation, gpt_model_name
        except requests.exceptions.RequestException as e:
            eval_logger.warning(f"Request failed on attempt {attempt + 1}: {e}")
            time.sleep(wait_time)
            if attempt == max_retries - 1:
                eval_logger.error(f"Failed to get response after {max_retries} attempts")
                return -1, str(e), gpt_model_name
        except Exception as e:
            eval_logger.error(f"Error on attempt {attempt + 1}: {e}")
            return -1, str(e), gpt_model_name


def image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


_images = {}

dataset = None


def livebench_doc_to_visual(doc):
    img_list = [image.convert("RGB") for image in doc["images"]]
    return img_list


def livebench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    return f"{pre_prompt}{doc['question']}{post_prompt}"


SUBTASKS = ("Basic Understanding", "Contextual Analysis", "Deeper Implications", "Broader Implications", "Further Insights")


def livebench_process_results_for_name(doc, results, model, eval_name):
    base64_images = [image_to_base64(image) for image in livebench_doc_to_visual(doc)]
    subtask = doc["subtask"]
    criteria = doc["criteria"]
    if subtask not in SUBTASKS:
        subtask = "further insights"
    if not results or results[0] == "":
        return {eval_name: {"rating": 0, "explanation": "No response", "model_name": "N/A", "subtask": subtask}}
    rating, explanation, model_name = get_chat_response(gpt_model_name=model, base64_images=base64_images, question=doc["question"], ground_truth_answer=doc["answer"], answer=results[0] if results else "", criteria=criteria)
    if rating >= 0:
        return {eval_name: {"rating": rating, "explanation": explanation, "model_name": model_name, "subtask": subtask, "id": doc["id"]}}
    else:
        return {eval_name: {"rating": -1, "explanation": explanation, "model_name": "N/A", "subtask": subtask, "id": doc["id"]}}


def livebench_process_results_4o(doc, results):
    return livebench_process_results_for_name(doc, results, "gpt-4o", "gpt4_eval_score")


def livebench_process_results_4o_mini(doc, results):
    return livebench_process_results_for_name(doc, results, "gpt-4o-mini", "gpt4_eval_score_mini")


def livebench_process_results(doc, results):
    res = livebench_process_results_4o(doc, results)
    if EVAL_WITH_MINI:
        res.update(livebench_process_results_4o_mini(doc, results))
    return res


def livebench_aggregate_results(results):
    sum_score, count = 0, 0
    score = {}
    for subtask in SUBTASKS:
        score[subtask] = []
    for result in results:
        if result["rating"] == -1:
            continue
        sum_score += result["rating"] / 10
        count += 1
        subtask = result["subtask"]
        if subtask not in SUBTASKS:
            subtask = "Further Insights"
        score[result["subtask"]].append(result["rating"] / 10)
    res = [(subtask, len(score[subtask]), np.mean(score[subtask]) * 100) for subtask in SUBTASKS]
    res.append(("Total", count, sum_score / count * 100))
    res = pd.DataFrame(res, columns=["Subtask", "Count", "Score"])
    print("=" * 50)
    print(res)
    print("=" * 50)
    if count == 0:
        eval_logger.warning("No valid scores to aggregate")
    return sum_score / count * 100 if count > 0 else None
