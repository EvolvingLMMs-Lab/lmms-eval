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

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

dir_name = os.path.dirname(os.path.abspath(__file__))


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

    return {"submission": {"index": doc["index"], "prediction": parsed_preds}}


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
