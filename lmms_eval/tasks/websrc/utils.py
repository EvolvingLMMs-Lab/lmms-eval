from collections import defaultdict
import re
import ast
import base64
import io
import random
import numpy as np
import os
import json
import logging
from PIL import Image

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

lmms_logger = logging.getLogger("lmms-eval")

OPEN_ENDED_PROMPT = "Answer the question using a single word or phrase."


def construct_prompt(doc):
    question = doc["question"]
    question = f"{OPEN_ENDED_PROMPT}\n{question}"
    return question


def websrc_doc_to_text(doc):
    question = construct_prompt(doc)
    return question


def websrc_doc_to_visual(doc):
    img_bs64 = doc["image"]
    img = Image.open(io.BytesIO(base64.b64decode(img_bs64)))
    del doc['image']
    return [img]


def websrc_process_results(doc, results):
    pred = results[0]
    parsed_pred = pred
    id = doc["page_id"]
    websrc_ans = {"id": id, "domain": doc['domain'], "answer": doc["answer"], "parsed_pred": parsed_pred}
    return {
        "websrc_squad_f1": websrc_ans,
        "submission": {
            id: pred,
        },
    }


def websrc_test_aggregate_results_for_submission(results, args):
    path = generate_submission_file("websrc_test_for_submission.json", args)
    with open(path, "w") as f:
        json.dump(results, f)
    lmms_logger.info(f"Results saved to {path}.")


def websrc_aggregate_results(results):
    evaluation_result = {}

    # Group results by domain
    subset_to_eval_samples = defaultdict(list)
    for result in results:
        subset_to_eval_samples[result["domain"]].append(result)

    # Evaluate each domain
    for subset, sub_eval_samples in subset_to_eval_samples.items():
        judge_dict, metric_dict = evaluate_websrc(sub_eval_samples)
        metric_dict.update({"num_example": len(sub_eval_samples)})
        evaluation_result[subset] = metric_dict

    # Aggregate results for all domains
    printable_results = {}
    for domain in DOMAINS:
        if domain not in evaluation_result:
            continue
        printable_results[domain] = {
            "num": int(evaluation_result[domain]["num_example"]),
            "f1": round(evaluation_result[domain]["f1"], 3),
        }
    all_ins_f1 = np.sum([cat_results["f1"] * cat_results["num_example"] for cat_results in evaluation_result.values()]) / sum(
        [cat_results["num_example"] for cat_results in evaluation_result.values()]
    )
    printable_results["Overall"] = {
        "num": sum([cat_results["num_example"] for cat_results in evaluation_result.values()]),
        "f1": round(all_ins_f1, 3),
    }
    print(printable_results)
    return printable_results["Overall"]["f1"]


##################
# Helper functions written by official MMMU repo.
##################
DOMAINS = [
    'auto',
    'book',
    'camera',
    'game',
    'jobs',
    'movie',
    'phone',
    'restaurant',
    'sports',
    'university',
    'hotel',
]


def evaluate_websrc(samples):

    def _normalize_str(string):
        # lower it
        string = string.lower()

        # strip non-alphanumeric characters
        string = re.sub(r"[^a-zA-Z0-9]", "", string)

        # strip leading and trailing whitespaces
        string = string.strip()
        
        return string

    judge_list = []
    for sample in samples:
        gold_i = set(_normalize_str(sample["answer"]))
        pred_i = set(_normalize_str( sample["parsed_pred"]))
        if len(pred_i) == 0:
            judge_list.append(0.0)
            continue

        comm_i = gold_i.intersection(pred_i)
        prec_i = len(comm_i) / len(pred_i)
        rec_i = len(comm_i) / len(gold_i)
        f1_i = 2 * prec_i * rec_i / (prec_i + rec_i) if prec_i + rec_i > 0 else 0
        judge_list.append(f1_i)

    f1 = np.mean(judge_list)
    return judge_list, {"f1": f1}
