from lmms_eval.tasks._task_utils.reasoning_utils import (
    make_reasoning_doc_to_messages,
    make_reasoning_process_results,
)
from lmms_eval.tasks.chartqa.utils import chartqa_doc_to_text as _chartqa_doc_to_text
from lmms_eval.tasks.chartqa.utils import chartqa_doc_to_visual


def chartqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {"pre_prompt": "", "post_prompt": ""}
    return _chartqa_doc_to_text(doc, kwargs)


chartqa_reasoning_doc_to_messages = make_reasoning_doc_to_messages(chartqa_doc_to_visual, chartqa_doc_to_text)

_base_process = make_reasoning_process_results("chartqa", chartqa_doc_to_text, extra_info_fn=lambda doc: {"type": doc["type"]})


def chartqa_reasoning_process_results(doc, results):
    doc_type = doc["type"]
    base = _base_process(doc, results)
    acc_score = base["acc_score"]
    format_score = base["format_score"]

    return_dict = {"relaxed_overall_acc_score": acc_score, "relaxed_overall_format_score": format_score}
    if doc_type == "human_test":
        return_dict["relaxed_human_split_acc_score"] = acc_score
        return_dict["relaxed_human_split_format_score"] = format_score
    else:
        return_dict["relaxed_augmented_split_acc_score"] = acc_score
        return_dict["relaxed_augmented_split_format_score"] = format_score
    return return_dict
