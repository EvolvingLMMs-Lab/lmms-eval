import re
import os
import json
import yaml
import pathlib

import datetime
import statistics

from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

from loguru import logger as eval_logger


def textvqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def textvqa_process_results(doc, result):
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
        accuracy = statistics.mean(gtAcc)

    return {
        "exact_match": accuracy,
        "submission": {
            "question_id": doc["question_id"],
            "answer": resAns,
        },
    }


def textvqa_doc_to_text(doc, model_specific_prompt_kwargs=None):
    pre_prompt = ""
    post_post = ""
    ocr_ref = ""
    if model_specific_prompt_kwargs:
        if "pre_prompt" in model_specific_prompt_kwargs:
            pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
        if "post_prompt" in model_specific_prompt_kwargs:
            post_prompt = model_specific_prompt_kwargs["post_prompt"]
        if "ocr" in model_specific_prompt_kwargs and model_specific_prompt_kwargs["ocr"]:
            ocr_ref = f"\nReference OCR token: {', '.join(doc['ocr_tokens'])}"
    return f"{pre_prompt}{doc['question'].capitalize()}{ocr_ref}{post_prompt}"


def textvqa_aggregate_submissions(results, args):
    now_date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    path = generate_submission_file(f"textvqa_submission_{now_date_time}.json", args)
    with open(path, "w") as f:
        json.dump(results, f)
    # print(f"Submission file saved to {path}")
    eval_logger.info(f"Submission file saved to {path}")
