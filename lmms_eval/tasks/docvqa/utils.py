import json
import os


def docvqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def docvqa_doc_to_text(doc, mdoel_specific_prompt_kwargs):
    question = doc["question"]
    pre_prompt = mdoel_specific_prompt_kwargs["pre_prompt"]
    post_prompt = mdoel_specific_prompt_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"


def docvqa_test_process_results(doc, results):
    pred = results[0]
    questionId = doc["questionId"]
    return {"anls": {"questionId": int(questionId), "answer": pred}}


def docvqa_test_aggregate_results(results):
    # save results as json
    os.makedirs("./submissions", exist_ok=True)
    with open("./submissions/docvqa_test_for_submission.json", "w") as f:
        json.dump(results, f)
    return -1
