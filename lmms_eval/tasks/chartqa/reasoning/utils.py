from lmms_eval.tasks._task_utils.reasoning_utils import (
    compute_score,
    make_reasoning_doc_to_messages,
)
from lmms_eval.tasks.chartqa.utils import chartqa_doc_to_text as _chartqa_doc_to_text
from lmms_eval.tasks.chartqa.utils import chartqa_doc_to_visual


def chartqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {"pre_prompt": "", "post_prompt": ""}
    return _chartqa_doc_to_text(doc, kwargs)


chartqa_reasoning_doc_to_messages = make_reasoning_doc_to_messages(chartqa_doc_to_visual, chartqa_doc_to_text)


def chartqa_reasoning_process_results(doc, results):
    doc_type = doc["type"]
    question = chartqa_doc_to_text(doc, None)
    ground_truth = doc["answer"]
    extra_info = {"question": question, "type": doc_type}

    acc_score = 0
    format_score = 0
    for pred in results:
        score_dict = compute_score(data_source="chartqa", solution_str=pred.strip(), ground_truth=ground_truth, extra_info=extra_info)
        acc_score += score_dict["acc_score"]
        format_score += score_dict.get("format_reward_score", 0.0)

    n = len(results) or 1
    acc_score /= n
    format_score /= n

    return_dict = {"relaxed_overall_acc_score": acc_score, "relaxed_overall_format_score": format_score}
    if doc_type == "human_test":
        return_dict["relaxed_human_split_acc_score"] = acc_score
        return_dict["relaxed_human_split_format_score"] = format_score
    else:
        return_dict["relaxed_augmented_split_acc_score"] = acc_score
        return_dict["relaxed_augmented_split_format_score"] = format_score
    return return_dict
