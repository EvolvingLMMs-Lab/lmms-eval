from lmms_eval.tasks._task_utils.reasoning_utils import (
    extract_anwser_tag,
    format_reward,
    make_reasoning_doc_to_messages,
)
from lmms_eval.tasks.ocrbench.utils import ocrbench_process_results


def ocrbench_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def ocrbench_doc_to_text(doc, lmms_eval_specific_kwargs):
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") if lmms_eval_specific_kwargs else ""
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "") if lmms_eval_specific_kwargs else ""
    question = doc["question"].strip()
    return f"{pre_prompt}{question}{post_prompt}"


ocrbench_reasoning_doc_to_messages = make_reasoning_doc_to_messages(ocrbench_doc_to_visual, ocrbench_doc_to_text)


def ocrbench_reasoning_process_results(doc, results):
    acc_score = 0
    format_score = 0
    for pred in results:
        format_score += format_reward(pred)
        extracted = extract_anwser_tag(pred).strip()
        score_dict = ocrbench_process_results(doc, [extracted])
        acc_score += score_dict["ocrbench_accuracy"]["score"]

    n = len(results) or 1
    return {"ocrbench_accuracy_acc_score": acc_score / n, "ocrbench_accuracy_format_score": format_score / n}
