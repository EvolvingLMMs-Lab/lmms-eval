import base64
import json
import os
import re
import time
from io import BytesIO
from pathlib import Path

import requests
import yaml

from lmms_eval.llm_judge import ServerConfig, get_server
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file


def doc_to_visual(doc):
    if doc["image"] is None:
        return []
    return [doc["image"].convert("RGB")]


from loguru import logger as eval_logger

# Assuming the config is loaded similarly as in d170_en/utils.py
with open(Path(__file__).parent / "dc100_en.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
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
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
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
    question = doc["question"]
    answer = doc["answer"]  # Ground truth answer
    
    # Define custom prompt for DC100 EN evaluation
    custom_prompt = """You are evaluating whether a model's response correctly describes an image.

The model should provide a correct and comprehensive description of the image including:
- Object/scene appearance, position, pose, action, shape
- Contents in the background
- Overall accuracy of the description

Do not penalize for:
- Appropriateness or sensitive descriptors (e.g., "middle-aged western man")
- Minor formatting differences

Score 1 if the prediction correctly and comprehensively describes the key elements in the image.
Score 0 if the prediction is incorrect, incomplete, or misses important elements.

Return only "1" or "0" with no additional text or formatting."""
    
    try:
        # Use the llm_judge API for binary evaluation
        result = server.evaluate_binary(
            question=question,
            answer=str(answer),
            prediction=prediction,
            output_format="1/0",
            custom_prompt=custom_prompt
        )
        
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
