import json
import os
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import openai
import requests
import yaml
from loguru import logger as eval_logger
from openai import OpenAI

from lmms_eval.llm_judge import ServerConfig, get_server

NUM_SECONDS_TO_SLEEP = 5

LLAVA_W_METRICS = ["gpt_eval_llava_conv", "gpt_eval_llava_detail", "gpt_eval_llava_complex"]

rule_dict = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rule.json"), "r"))

with open(Path(__file__).parent / "llava-in-the-wild.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

API_TYPE = os.getenv("API_TYPE", "openai")
MODEL_VERSION = os.getenv("MODEL_VERSION", "gpt-4o-2024-08-06")

server_config = ServerConfig(
    model_name=MODEL_VERSION,
)
server = get_server(server_name=API_TYPE, config=server_config)


def llava_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def llava_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    return f"{pre_prompt}{doc['question']}{post_prompt}"


def llava_process_results(doc, result):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case coco_bleu), value: metric value
    """
    try:
        question = doc.get("question", "")
        ans1 = doc.get("gpt_answer", "")
        ans2 = result[0] if result else ""
        captions = doc.get("caption", [])
        context = "\n".join(captions) if isinstance(captions, list) else captions
        category = "llava_bench_" + doc.get("category", "")
        rule = rule_dict.get(category, {})
        prompt = rule.get("prompt", "")
        role = rule.get("role", "user")
        content = f"[Context]\n{context}\n\n" f"[Question]\n{question}\n\n" f"[{role} 1]\n{ans1}\n\n[End of {role} 1]\n\n" f"[{role} 2]\n{ans2}\n\n[End of {role} 2]\n\n" f"[System]\n{prompt}\n\n"

        result = server.evaluate_comparative(question=question, response1=ans1, response2=ans2, context=context, custom_prompt=content, score_range=(1, 10))

        review = result["raw_response"]
        model_name = result["model"]
        scores = list(result["scores"])
    except Exception as e:
        eval_logger.error(f"Error for Question ID: {doc.get('question_id', 'Unknown')}: {e}")
        review = "Failed to Get a Proper Review."
        model_name = "Failed Request"
        scores = [-1, -1]

    metric = f"gpt_eval_llava_{doc.get('category', 'all')}"
    category_review_dict = {"question": question, "ans1": ans1, "ans2": ans2, "context": context, "category": category, "review": review, "scores": scores, "eval_model": model_name, "content": content}

    non_category_review_dict = deepcopy(category_review_dict)
    non_category_review_dict["scores"] = [-999, -999]

    data_dict = {}
    for m in LLAVA_W_METRICS:
        if m == metric:
            data_dict[m] = category_review_dict
        else:
            data_dict[m] = non_category_review_dict
    data_dict["gpt_eval_llava_all"] = category_review_dict

    # return {"gpt_eval_llava_all": review_dict}
    return data_dict


def llava_conv_aggregation(results):
    return llava_aggregation(results, "conv")


def llava_complex_aggregation(results):
    return llava_aggregation(results, "complex")


def llava_detail_aggregation(results):
    return llava_aggregation(results, "detail")


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
