import re
import os
import json
import yaml
import pathlib
import logging
import datetime
import statistics

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor

eval_logger = logging.getLogger("lmms-eval")


def ok_vqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def ok_vqa_process_results(doc, result):
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


def ok_vqa_doc_to_text(doc, model_specific_prompt_kwargs=None):
    question = doc["question"]
    if model_specific_prompt_kwargs is None:
        model_specific_prompt_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in model_specific_prompt_kwargs:
        pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
    if "post_prompt" in model_specific_prompt_kwargs:
        post_prompt = model_specific_prompt_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"


def ok_vqa_aggreate_submissions(results, args):
    now_date_time = datetime.datetime.now().strftime("%Y-%m%d-%H%M-%S")
    file = f"ok_vqa-test-submission-{now_date_time}.json"
    path = generate_submission_file(file, args)
    with open(path, "w") as f:
        json.dump(results, f)
    print(f"Submission file saved to {path}")
