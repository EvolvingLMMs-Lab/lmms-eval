import logging
import re
from PIL import Image
import numpy as np
import io
import pandas as pd
from collections import defaultdict
from lmms_eval.filters.extraction import ExtendedRegexFilter
from lmms_eval.filters.transformation import MapFilter
import re

eval_logger = logging.getLogger("lmms-eval")


def msr_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"].strip()
    if "pre_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["pre_prompt"] != "":
        question = f"{lmms_eval_specific_kwargs['pre_prompt']}{question}"
    if "post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["post_prompt"] != "":
        question = f"{question}{lmms_eval_specific_kwargs['post_prompt']}"
    return question


def msr_doc_to_visual(doc):
    # image_list = [image.convert("RGB") for image in doc["images"]]
    image_list = []
    for img_data in doc["images"]:
        image = Image.open(io.BytesIO(img_data))
        image = image.convert("RGB")
        image_list.append(image)
    return image_list




def extract_single_choice_with_word_boundary(pred, gt):
    pattern_1 = r'``([^`]*)``'
    match = re.search(pattern_1, pred)
    if match:
        pred = match.group(1)  

    pattern_2 = r'`([^`]*)`'
    match = re.search(pattern_2, pred)
    if match:
        pred = match.group(1)  

    pattern_add = r'\{([^}]*)\}'
    match = re.search(pattern_add, pred)
    if match:
        pred = match.group(1)  

    pattern_3 = r'\b[A-D]\b(?!\s[a-zA-Z])'
    match = re.search(pattern_3, pred)
    if match:
        pred = match.group()  
    else:
        return None 

    answer = gt.lower().replace("\n", " ").strip()
    predict = pred.lower().replace("\n", " ").strip()
    try:
        if answer == predict[0]:
            return 1.0
        elif predict[0] == "(" and answer == predict[1]:
            return 1.0
        elif predict[0:7] == "option " and answer == predict[7]:
            return 1.0
        elif predict[0:14] == "the answer is " and answer == predict[14]:
            return 1.0
    except Exception as e:
        return 0.0
    return 0.0



def msr_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name, value: metric value
    """
    pred = results[0]
    gt = doc["answer"]

    score = extract_single_choice_with_word_boundary(pred, gt)
    category = doc["question_type"]
    l2_category = doc["question_type"]
    if score is None:
        return {category: {"question_id": doc["id"], "l2_category": l2_category, "score": 0, "note": "can not find anwser"}, "average": {"question_id": doc["id"], "l2_category": l2_category, "score": 0, "note": "can not find anwser"}}
    return {category: {"question_id": doc["id"], "l2_category": l2_category, "score": score}, "average": {"question_id": doc["id"], "l2_category": l2_category, "score": score}}


def msr_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    l2_category_scores = defaultdict(list)
    for result in results:
        score = result["score"]
        l2_category = result["l2_category"]
        l2_category_scores[l2_category].append(score)

    l2_category_avg_score = {}
    for l2_category, scores in l2_category_scores.items():
        avg_score = sum(scores) / len(scores)
        l2_category_avg_score[l2_category] = avg_score
        eval_logger.info(f"{l2_category}: {avg_score:.2f}")

    all_scores = [score for scores in l2_category_scores.values() for score in scores]
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    return avg_score

