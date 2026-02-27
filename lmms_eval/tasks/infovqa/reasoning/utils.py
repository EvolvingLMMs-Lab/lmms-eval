from lmms_eval.api.metrics import anls
from lmms_eval.tasks._task_utils.reasoning_utils import (
    extract_anwser_tag,
    format_reward,
    make_reasoning_doc_to_messages,
)


def infovqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def infovqa_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") if lmms_eval_specific_kwargs else ""
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "") if lmms_eval_specific_kwargs else ""
    return f"{pre_prompt}{question}{post_prompt}"


infovqa_reasoning_doc_to_messages = make_reasoning_doc_to_messages(infovqa_doc_to_visual, infovqa_doc_to_text)


def infovqa_reasoning_process_results(doc, results):
    ground_truth = doc["answers"]
    acc_score = 0
    format_score = 0
    for pred in results:
        extracted = extract_anwser_tag(pred).strip()
        anls_score = anls(ground_truth, [extracted])
        acc_score += anls_score["anls"]
        format_score += format_reward(pred)

    n = len(results) or 1
    return {
        "anls": acc_score / n,
        "anls_acc_score": acc_score / n,
        "anls_format_score": format_score / n,
    }
