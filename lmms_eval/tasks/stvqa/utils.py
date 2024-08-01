import json
from loguru import logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file


def stvqa_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"


def stvqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def stvqa_process_results(doc, results):
    answer = results[0]
    return {"submission": {"question_id": int(doc["question_id"]), "answer": answer}}


def stvqa_aggregate_submissions(results, args):
    file = generate_submission_file("stvqa_test_for_submission.json", args)
    with open(file, "w") as f:
        json.dump(results, f)
    logger.info(f"Results saved to {file}")
