import json
import logging
import os
import requests
import numpy as np
import openai
from openai import OpenAI
import time
import yaml
from pathlib import Path

eval_logger = logging.getLogger("lmms-eval")
NUM_SECONDS_TO_SLEEP = 0.5

rule_dict = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rule.json"), "r"))

with open(Path(__file__).parent / "llava-in-the-wild.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
GPT_EVAL_MODEL_NAME = config["metadata"]["gpt_eval_model_name"]


def get_eval(content: str, max_tokens: int):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    messages = [{"role": "system", "content": "You are a helpful and precise assistant for checking the quality of the answer."}, {"role": "user", "content": content}]

    payload = {"model": GPT_EVAL_MODEL_NAME, "messages": messages, "temperature": 0.2, "max_tokens": max_tokens}

    while True:
        try:
            response = requests.post(API_URL, headers=headers, json=payload)
            response.raise_for_status()
            response_data = response.json()

            content = response_data["choices"][0]["message"]["content"].strip()
            if content != "":
                return content, response_data["model"]

        except Exception as e:
            eval_logger.info(f"Error in response : {response.json()['error']['message']}")
            if "Rate limit" in str(e):
                eval_logger.info("Sleeping due to rate limit...")
                time.sleep(NUM_SECONDS_TO_SLEEP)
    return "", ""


def parse_score(review):
    try:
        score_pair = review.split("\n")[0]
        score_pair = score_pair.replace(",", " ")
        sp = score_pair.split(" ")
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            eval_logger.debug("error", review)
            return [-1, -1]
    except Exception as e:
        eval_logger.debug(e)
        eval_logger.debug("error", review)
        return [-1, -1]


def llava_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def llava_doc_to_text(doc):
    question = doc["question"]
    return question


def llava_process_results(doc, result):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case coco_bleu), value: metric value
    """
    question = doc["question"]
    ans1 = doc["gpt_answer"]
    ans2 = result[0]
    if isinstance(doc["caption"], list):
        context = "\n".join(doc["caption"])
    else:
        context = doc["caption"]
    category = "llava_bench_" + doc["category"]
    rule = rule_dict[category]
    prompt = rule["prompt"]
    role = rule["role"]
    content = f"[Context]\n{context}\n\n" f"[Question]\n{question}\n\n" f"[{role} 1]\n{ans1}\n\n[End of {role} 1]\n\n" f"[{role} 2]\n{ans2}\n\n[End of {role} 2]\n\n" f"[System]\n{prompt}\n\n"

    review, model_name = get_eval(content, 1024)
    scores = parse_score(review)
    metric = f"gpt_eval_llava_{doc['category']}"
    review_dict = {"question": question, "ans1": ans1, "ans2": ans2, "context": context, "category": category, "review": review, "scores": scores, "eval_model": model_name}

    return {metric: review_dict, "gpt_eval_llava_all": review_dict}


def llava_aggregation(results):
    scores = []
    category = results[0]["category"]
    for result in results:
        scores.append(result["scores"])

    stats = np.asarray(scores).mean(0).tolist()
    stats = [round(x, 3) for x in stats]
    eval_logger.info(f"Model/GPT4 Score for {category}: {stats[1] / stats[0] * 100:.1f}%")
    eval_logger.info(f"GPT4 Score for {category}: {stats[0] * 10:.1f}%")
    eval_logger.info(f"Model Score for {category}: {stats[1] * 10:.1f}%")
    # TODO: For KC, Please make the logging information more clear. e.g. GPT4 Score: 0.8, Model Score: 0.7...
    eval_logger.info("=========================")
    return round(stats[1] / stats[0] * 100, 1)
