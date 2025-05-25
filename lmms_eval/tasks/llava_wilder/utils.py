import base64
import os
import time
from copy import deepcopy
from http import HTTPStatus
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
import yaml

# Set up a logger
from loguru import logger as eval_logger

from lmms_eval.api.judge_config_helper import create_judge
from lmms_eval.api.judge_utils import ResponseParser

judge_rules = "We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image shown to you. \nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance. Assume assistant 1 always receive a score of 10 and is the correct answer.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.\nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."

with open(Path(__file__).parent / "_default_template_wilder_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

# Initialize judge for LLaVA Wilder evaluations using environment-based configuration
wilder_judge = create_judge(yaml_config=config)


def image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def parse_score(review):
    """Use unified response parser"""
    scores = ResponseParser.parse_comparative_response(review)
    return list(scores)


def llava_process_results(doc, result):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case coco_bleu), value: metric value
    """
    try:
        question = doc.get("Question", "")
        ans1 = doc.get("Answer", "")
        ans2 = result[0] if result else ""
        content = f"[Question]\n{question}\n\n" + f"[Assistant 1]\n{ans1}\n\n[End of Assistant 1]\n\n" + f"[Assistant 2]\n{ans2}\n\n[End of Assistant 2]\n\n" f"[System]\n{judge_rules}\n\n"
        visuals = llava_doc_to_visual(doc)[0]
        base64_image = image_to_base64(visuals)
        # Use unified judge API with image support
        result = wilder_judge.evaluate_comparative(question=question, response1=ans1, response2=ans2, custom_prompt=content, images=[base64_image.encode()], score_range=(1, 10))  # Convert to bytes

        review = result["raw_response"]
        model_name = result["model"]
        scores = list(result["scores"])
    except Exception as e:
        eval_logger.error(f"Error for Question ID: {doc.get('question_id', 'Unknown')}: {e}")
        eval_logger.error(f"Try to set up --process_with_media=True to process the dataset with media to avoid image missing error.")
        review = "Failed to Get a Proper Review."
        model_name = "Failed Request"
        scores = [-1, -1]

    data_dict = {"question": question, "ans1": ans1, "ans2": ans2, "review": review, "scores": scores, "eval_model": model_name, "content": content}
    # return {"gpt_eval_llava_all": review_dict}
    return {"gpt_eval_llava_all": data_dict}


def llava_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def llava_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    return f"{pre_prompt}{doc['Question']}{post_prompt}"


def llava_all_aggregation(results):
    return llava_aggregation(results, "all")


def llava_aggregation(results, category):
    try:
        scores = []
        for result in results:
            if -999 in result["scores"]:
                continue
            scores.append(result["scores"])

        stats = np.asarray(scores).mean(0).tolist()
        stats = [round(x, 3) for x in stats]
        # gpt4_score_percentage = stats[0] * 10
        # model_score_percentage = stats[1] * 10
        # eval_logger.info(f"Category: {category}")
        # eval_logger.info(f"GPT4 Score: {gpt4_score_percentage:.1f}%")
        # eval_logger.info(f"Model Score: {model_score_percentage:.1f}%")
        # eval_logger.info("=========================")
        return round(stats[1] / stats[0] * 100, 1)
    except Exception as e:
        eval_logger.info(f"Error in llava_aggregation: {e}, and in category: {category}")
        return None
