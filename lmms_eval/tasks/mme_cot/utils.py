import base64
import datetime
import io
import json
import os
import string
from collections import defaultdict

import pandas as pd
from loguru import logger as eval_logger
from openai import AzureOpenAI, OpenAI
from PIL import Image

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

dir_name = os.path.dirname(os.path.abspath(__file__))

API_TYPE = os.getenv("API_TYPE", "openai")
if API_TYPE == "openai":
    API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    client = OpenAI(api_key=API_KEY)
    gpt_model = config["metadata"]["gpt_eval_model_name"]

elif API_TYPE == "azure":
    API_URL = os.getenv("AZURE_ENDPOINT", "https://api.cognitive.microsoft.com/sts/v1.0/issueToken")
    API_KEY = os.getenv("AZURE_API_KEY", "YOUR_API_KEY")
    API_VERSION = os.getenv("AZURE_API_VERSION", "2023-07-01-preview")
    client = AzureOpenAI(azure_endpoint=API_URL, api_version=API_VERSION, api_key=API_KEY)
    gpt_model = os.getenv("MODEL_VERSION", "gpt-4o-2024-11-20")


def get_chat_response(prompt, max_token=256, retry=5):
    messages = [
        {"role": "user", "content": prompt},
    ]
    for i in range(retry):
        try:
            completion = client.chat.completions.create(model=gpt_model, messages=messages, temperature=0.5 * i, max_tokens=max_token)
            prediction = completion.choices[0].message.content.strip()
            if prediction.lower() == "yes" or prediction.lower() == "no":
                return prediction
        except Exception as e:
            eval_logger.error(e)
    return "no"


def build_mmecot_gpt4_prompt(question_data):
    prompt = """You are given a question, the solution and the correct answer. Please determine if the solution matches the correct answer.
Focus only on the mathematical or semantic correctness of the content. Ignore any differences in formatting, such as LaTeX syntax, symbols, styles, or additional wrappers (e.g., \boxed, $...$, or similar). Compare only the core mathematical or textual meaning of the solution and the correct answer.
The process or reasoning leading to the Solution is irrelevant, ONLY the correctness of the result matters.
Return only "Yes" if the solution is correct or "No" if it is incorrect.
Only return "Yes" or "No" with no additional text or formatting.

Question: 
{question}
--------------------------------
Correct Answer:
{answer}
--------------------------------
Solution: 
{solution}
--------------------------------
"""
    question = question_data["question"]
    answer = question_data["answer"]
    response = str(question_data["response"])
    prompt = prompt.format(question=question, answer=answer, solution=response)
    return prompt


def mmecot_doc_to_visual(doc):
    visual_list = []
    for image in doc["image"]:
        base64_image = image
        image = Image.open(io.BytesIO(base64.b64decode(base64_image))).convert("RGB")
        visual_list.append(image)
    return visual_list


def mmecot_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    # Get base prompt from question
    prompt = doc["question"].strip()

    # Apply pre_prompt and post_prompt if provided
    if lmms_eval_specific_kwargs:
        if "pre_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["pre_prompt"] != "":
            prompt = f"{lmms_eval_specific_kwargs['pre_prompt']}{prompt}"
        if "post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["post_prompt"] != "":
            prompt = f"{prompt}{lmms_eval_specific_kwargs['post_prompt']}"

    # Add options if available
    options = {cand: doc[cand] for cand in string.ascii_uppercase if cand in doc and not pd.isna(doc[cand])}
    if options:
        prompt = prompt + "\n" + "\n".join([f"{key}. {item}" for key, item in options.items()])

    if lmms_eval_specific_kwargs["postfix_type"] == "direct":
        prompt += "\nPlease directly provide the final answer without any other output."
    elif lmms_eval_specific_kwargs["postfix_type"] == "cot":
        prompt += "\nPlease generate a step by step answer, include all your intermediate reasoning process, and provide the final answer at the end."

    return prompt


def mmecot_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case mme score), value: metric value
    """
    parsed_preds = []
    for pred in results:
        parsed_preds.append(pred)

    prediction = results[0].strip()
    # Build the prompt for GPT-4o evaluation
    question_data = {"index": doc.get("index", "unknown"), "question": doc["question"], "answer": doc["answer"], "response": prediction}

    # Build the prompt and get GPT-4o's judgment
    prompt = build_mmecot_gpt4_prompt(question_data)
    try:
        completion = get_chat_response(prompt)
        if completion.lower() == "yes" or completion.lower() == "no":
            judge_result = 1 if completion.lower() == "yes" else 0
        else:
            eval_logger.error(f"Invalid response: {completion}")
            judge_result = 0
    except Exception as e:
        eval_logger.error(f"Error getting chat response: {e}")
        judge_result = 0

    return {"submission": {"index": doc["index"], "prediction": parsed_preds}, "llm_as_judge_eval": judge_result}


def mmecot_reasoning_aggregate_results(results, args):
    path = generate_submission_file("mmecot_reasoning_test_for_submission.json", args)
    with open(path, "w") as f:
        json.dump(results, f)
    eval_logger.info(f"Results saved to {path}.")


def mmecot_direct_aggregate_results(results, args):
    path = generate_submission_file("mmecot_direct_test_for_submission.json", args)
    with open(path, "w") as f:
        json.dump(results, f)
    eval_logger.info(f"Results saved to {path}.")
