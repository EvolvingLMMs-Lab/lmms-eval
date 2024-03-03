import base64
import requests
import re
import logging
import time
import os
import yaml
import json
from pathlib import Path
from io import BytesIO

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file


def doc_to_visual(doc):
    if doc["image"] is None:
        return []
    return [doc["image"].convert("RGB")]


eval_logger = logging.getLogger("lmms-eval")

# Assuming the config is loaded similarly as in d170_en/utils.py
with open(Path(__file__).parent / "dc100_en.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
    config = yaml.safe_load("".join(safe_data))

API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
GPT_EVAL_MODEL_NAME = config["metadata"]["gpt_eval_model_name"]

EVALUATION_PROMPT_TEMPLATE_SIMPLE_V1 = """Text Caption: {caption}
From 0 to 100, how much do you rate for this Text Caption in terms of the correct and comprehensive description of the image?
Do not dominant the rating by a single attribute such as recognition correctness, but a overall rating on the object/scene appearance, position, pose, action, shape, etc., and contents in the background. 
Do not consider the appropriateness or sensitive descriptors, such as "middle-aged western man", judge based on if it has correct specifications of the object and scenes in image.
Provide a few lines for explanation and the rate number at last after "Final Score:"."""


def get_chat_response(base64_image, prompt, max_retries=5, wait_time=10):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": GPT_EVAL_MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    },
                ],
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.0,
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload)
            response.raise_for_status()
            response_data = response.json()
            return response_data["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            eval_logger.warning(f"Request failed on attempt {attempt+1}: {e}")
            time.sleep(wait_time)
            if attempt == max_retries - 1:
                eval_logger.error(f"Failed to get response after {max_retries} attempts")
                return ""
        except Exception as e:
            eval_logger.error(f"Error on attempt {attempt+1}: {e}")
            return ""


def image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def process_results(doc, results):
    prediction = results[0]
    question_id = doc["question_id"]
    image_path = doc["image"]
    base64_image = image_to_base64(image_path)
    prompt = EVALUATION_PROMPT_TEMPLATE_SIMPLE_V1.format(caption=prediction)
    try:
        response = get_chat_response(base64_image, prompt)
        score_value = re.search(r"Final Score: (\d+)", response)
        score = int(score_value.group(1)) if score_value else 0
    except Exception as e:
        eval_logger.error(f"Error for Question ID: {question_id}: {e}")
        response = ""
        score = 0

    return {
        "gpt_eval_info": {
            "question_id": question_id,
            "question": doc["question"],
            "model_caption": prediction,
            "explanation": response,
            "eval_model": GPT_EVAL_MODEL_NAME,
            "score": score,
            "prompt" : prompt
        },
        "gpt_eval_avg_score": {
            "score": score,
        },
    }


def dc100_en_aggregate_info(results, args):
    path = generate_submission_file("dc100_en_eval_info.json", args)
    with open(path, "w") as f:
        json.dump(results, f)
    eval_logger.info(f"Results saved to {path}.")


def dc100_en_aggregate_avg_score(results):
    total_score = 0
    for result in results:
        total_score += result["score"]
    avg_score = total_score / len(results)
    return avg_score
