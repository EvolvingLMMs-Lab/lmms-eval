import json
import os


from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

from loguru import logger


def docvqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def docvqa_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"


def docvqa_test_process_results(doc, results):
    pred = results[0]
    questionId = doc["questionId"]
    return {"anls": {"questionId": int(questionId), "answer": pred}, "submission": {"questionId": int(questionId), "answer": pred}}


def docvqa_test_aggregate_results(results, args):
    # save results as json
    path = generate_submission_file("docvqa_test_for_submission.json", args)
    with open(path, "w") as f:
        json.dump(results, f)
    logger.info(f"Results saved to {path}")
