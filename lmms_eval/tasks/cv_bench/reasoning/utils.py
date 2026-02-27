from lmms_eval.tasks._task_utils.reasoning_utils import (
    make_reasoning_doc_to_messages,
    make_reasoning_process_results,
)


def cv_bench_doc_to_visual(doc: dict) -> list:
    return [doc["image"].convert("RGB")]


def cv_bench_doc_to_text(doc: dict, lmms_eval_specific_kwargs=None) -> str:
    num_choices = len(doc["choices"])
    choice_letters = ", ".join([chr(65 + i) for i in range(num_choices)])
    prompt = lmms_eval_specific_kwargs.get("pre_prompt", "").format(choice_letters) if lmms_eval_specific_kwargs else ""
    prompt += "\n" + doc["prompt"]
    return prompt


cv_bench_reasoning_doc_to_messages = make_reasoning_doc_to_messages(cv_bench_doc_to_visual, cv_bench_doc_to_text)
cv_bench_reasoning_process_results = make_reasoning_process_results("cv_bench", cv_bench_doc_to_text, metrics_prefix="cv_bench_")
