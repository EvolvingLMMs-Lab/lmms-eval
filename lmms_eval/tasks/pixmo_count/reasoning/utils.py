import re

from lmms_eval.tasks._task_utils.reasoning_utils import make_reasoning_doc_to_messages
from lmms_eval.tasks.pixmo_count.utils import (
    pixmo_count_doc_to_text,
    pixmo_count_doc_to_visual,
)

pixmo_count_reasoning_doc_to_messages = make_reasoning_doc_to_messages(pixmo_count_doc_to_visual, pixmo_count_doc_to_text)


def pixmo_count_reasoning_process_results(doc, results):
    from lmms_eval.tasks._task_utils.reasoning_utils import compute_score

    question = pixmo_count_doc_to_text(doc, None)
    ground_truth = doc["answer"].strip()

    match = re.search(r"\d+", ground_truth)
    if match:
        ground_truth = match.group()

    extra_info = {"question": question}

    acc_score = 0
    format_score = 0
    for pred in results:
        score_dict = compute_score(data_source="pixmo_count", solution_str=pred.strip(), ground_truth=ground_truth, extra_info=extra_info)
        acc_score += score_dict["acc_score"]
        format_score += score_dict.get("format_reward_score", 0.0)

    n = len(results) or 1
    return {"acc_score": acc_score / n, "format_score": format_score / n}
