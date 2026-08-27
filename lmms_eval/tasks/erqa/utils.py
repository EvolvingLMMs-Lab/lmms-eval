import logging
from collections import defaultdict

from lmms_eval.tasks._task_utils.mcq_extract import extract_mcq_answer

eval_logger = logging.getLogger("lmms-eval")


def erqa_doc_to_text(doc: dict) -> str:
    return doc["question"]


def erqa_doc_to_visual(doc: dict) -> list:
    image_list = []
    for image in doc["images"]:
        if image is not None:
            image_list.append(image.convert("RGB"))
    return image_list


def erqa_doc_to_messages(doc: dict, lmms_eval_specific_kwargs=None) -> list[dict]:
    content = [{"type": "image", "url": image} for image in erqa_doc_to_visual(doc)]
    content.append({"type": "text", "text": erqa_doc_to_text(doc)})
    return [{"role": "user", "content": content}]


def erqa_process_results(doc, results):
    key_name = "erqa_acc"
    # extract grounded answer
    grounded_output = doc["answer"]
    response = results[0]

    # extract predicted answer
    pred_letter = extract_mcq_answer(response, choices=["A", "B", "C", "D"])
    flag = pred_letter == grounded_output

    omnispatial_submission = {"id": doc["question_id"], "gt_content": grounded_output, "pred": response, "sub_task": doc["question_type"], "is_correct": flag}
    return {key_name: omnispatial_submission}


def erqa_aggregate_results(results):
    sub_task_to_eval_samples = defaultdict(list)
    total_samples = len(results)
    total_correct = 0

    for sample in results:
        sub_task = sample["sub_task"]
        is_correct = sample["is_correct"]

        if is_correct:
            total_correct += 1
            sub_task_to_eval_samples[sub_task].append(1)
        else:
            sub_task_to_eval_samples[sub_task].append(0)

    accuracy = total_correct / total_samples if total_samples > 0 else 0
    sub_task_accuracies = {sub_task: sum(scores) / len(scores) for sub_task, scores in sub_task_to_eval_samples.items()}

    eval_logger.info("%-40s", "ERQA per-sub-task accuracy")
    eval_logger.info("-" * 40)
    for sub_task, acc in sub_task_accuracies.items():
        eval_logger.info("%-20s: %.4f", sub_task, acc)
    eval_logger.info("=" * 40)

    return accuracy
