import datetime
import json
import os
import pathlib
import re
import statistics

import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor


def vizwiz_vqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def vizwiz_vqa_process_results(doc, result):
    eval_ai_processor = EvalAIAnswerProcessor()
    assert len(result) == 1, f"The result should be a list of length 1, but got {len(result)}."
    resAns = eval_ai_processor(result[0])
    accuracy = 0

    if "answers" in doc and doc["answers"] is not None:
        gtAcc = []

        for i in range(len(doc["answers"])):
            doc["answers"][i] = eval_ai_processor(doc["answers"][i])

        for i in range(len(doc["answers"])):
            otherGTAns = [doc["answers"][j] for j in range(len(doc["answers"])) if i != j]
            matchingAns = [item for item in otherGTAns if item == resAns]
            acc = min(1, float(len(matchingAns)) / 3)
            gtAcc.append(acc)
        if gtAcc:
            accuracy = statistics.mean(gtAcc)
        else:
            accuracy = 0

    return {
        "exact_match": accuracy,
        "submission": {
            "image": f"{doc['question_id']}.jpg",
            "answer": resAns,
        },
    }


def vizwiz_vqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    if "post_prompt" in lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    text = f"{pre_prompt}{doc['question'].capitalize()}{post_prompt}"
    return text


def vizwiz_vqa_aggregate_submissions(results, args):
    now_date_time = datetime.datetime.now().strftime("%Y-%m%d-%H%M-%S")
    submission_file_name = f"vizwiz_vqa-test-submission-{now_date_time}.json"
    path = generate_submission_file(submission_file_name, args)
    with open(path, "w") as f:
        json.dump(results, f)
    print(f"Submission file saved to {path}")
