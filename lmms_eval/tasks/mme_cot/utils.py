import base64
import datetime
import io
import json
import os
import string
from collections import defaultdict

import pandas as pd
from loguru import logger as eval_logger
from PIL import Image

from lmms_eval.llm_judge import ServerConfig, get_server
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

dir_name = os.path.dirname(os.path.abspath(__file__))

# Initialize the judge server
API_TYPE = os.getenv("API_TYPE", "openai")
GPT_MODEL = os.getenv("MODEL_VERSION", "gpt-4o-2024-11-20")

server_config = ServerConfig(
    model_name=GPT_MODEL,
)
server = get_server(server_name=API_TYPE, config=server_config)


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
    question = doc["question"]
    answer = doc["answer"]

    # Define custom prompt for MME-CoT evaluation
    custom_prompt = """You are given a question, the solution and the correct answer. Please determine if the solution matches the correct answer.
Focus only on the mathematical or semantic correctness of the content. Ignore any differences in formatting, such as LaTeX syntax, symbols, styles, or additional wrappers (e.g., \boxed, $...$, or similar). Compare only the core mathematical or textual meaning of the solution and the correct answer.
The process or reasoning leading to the Solution is irrelevant, ONLY the correctness of the result matters.
Return only "Yes" if the solution is correct or "No" if it is incorrect.
Only return "Yes" or "No" with no additional text or formatting."""

    try:
        # Use the llm_judge API for binary evaluation
        result = server.evaluate_binary(question=question, answer=answer, prediction=prediction, output_format="yes/no", custom_prompt=custom_prompt)

        # Parse the result
        if result["success"]:
            judge_response = result["result"]
            judge_result = 1 if judge_response and judge_response.lower() == "yes" else 0
        else:
            eval_logger.error(f"Judge evaluation failed: {result.get('raw_response', 'Unknown error')}")
            judge_result = 0

    except Exception as e:
        eval_logger.error(f"Error getting judge response: {e}")
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
