import io
import base64
import logging
import re
import random 
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
Image.MAX_IMAGE_PIXELS = None 
from lmms_eval.filters.extraction import ExtendedRegexFilter
from lmms_eval.filters.transformation import MapFilter

eval_logger = logging.getLogger("lmms-eval")


def gmai_doc_to_visual(doc):
    image_data = base64.b64decode(doc['image'])
    image = Image.open(io.BytesIO(image_data))
    image = image.convert("RGB")
    return [image]


def gmai_doc_to_text(doc,lmms_eval_specific_kwargs=None):
    question = doc["question"].strip()
    question = question + "\nOptions:"
    dict_opt = {"A":"A","B":"B","C":"C","D":"D","E":"E"}
    for item in dict_opt:
        if doc.get(item):
            question = question + "\n" + f"{dict_opt[item]}: " + doc[item]
    if "pre_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["pre_prompt"] != "":
        question = f"{lmms_eval_specific_kwargs['pre_prompt']}{question}"
    if "post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["post_prompt"] != "":
        question = f"{question}{lmms_eval_specific_kwargs['post_prompt']}"
    return question

def gmai_parse_results(pred, valid_options=("a", "b", "c", "d", "e")):
    """
    Parses the model's prediction and extracts a single-choice answer from valid options.

    Args:
        pred (str): Model's prediction text.
        valid_options (tuple): Acceptable answer letters in lowercase.

    Returns:
        str: The parsed answer (e.g., 'a', 'b', 'c', 'd', 'e') or 'other' if invalid.
    """
    if not isinstance(pred, str):
        return "other"

    # Normalize
    pred = pred.lower().strip()

    # If 'answer:' appears anywhere, keep only what's after it
    answer_match = re.search(r"answer\s*:\s*(.*)", pred)
    if answer_match:
        pred = answer_match.group(1).strip()

    # Remove punctuation that could follow letters
    pred = re.sub(r"[.\)\]]", "", pred)

    # Regex to find the first valid letter (a–e)
    match = re.search(r"\b[a-e]\b", pred)
    if match:
        return match.group(0)

    # Fallback: if first char is a valid option
    if len(pred) > 0 and pred[0] in valid_options:
        return pred[0]

    return "other"

def gmai_process_results(doc,results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name, value: metric value
    """
    pred = results[0]
    gt = doc["answer"].lower().strip().replace(".", "")
    pred_ans = gmai_parse_results(pred)
    if pred_ans is None or pred_ans == "other":
        score = 0.0
    score = 1.0 if pred_ans == gt else 0.0
    return {"accuracy": {"question_id" : doc['index'], "score" : score}}


def gmai_aggregate_results(results):
    if not results:
        return 0
    question2scores = defaultdict(list)

    for result in results:
        qid = result.get("question_id")
        score = result.get("score", 0.0)
        if qid is not None:
            question2scores[qid].append(score)

    # Compute accuracy (average score)
    total_score = sum(sum(scores) for scores in question2scores.values())
    total_items = sum(len(scores) for scores in question2scores.values())
    accuracy = (total_score / total_items) * 100.0 if total_items > 0 else 0.0

    eval_logger.info(f"Overall Accuracy: {accuracy:.2f}")

    return  accuracy

