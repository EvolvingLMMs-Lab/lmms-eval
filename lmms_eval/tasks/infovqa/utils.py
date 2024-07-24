import json
import os


from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

from loguru import logger as eval_logger


def infovqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def infovqa_doc_to_text(doc, model_specific_prompt_kwargs):
    question = doc["question"]
    pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
    post_prompt = model_specific_prompt_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"


def infovqa_test_process_results(doc, results):
    pred = results[0]
    questionId = doc["questionId"]
    return {"submission": {"questionId": int(questionId), "answer": pred}}


def infovqa_test_aggregate_results(results, args):
    # save results as json
    file = generate_submission_file("infovqa_test_for_submission.json", args)
    with open(file, "w") as f:
        json.dump(results, f)
    eval_logger.info(f"Results saved to {file}")
