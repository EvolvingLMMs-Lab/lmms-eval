from lmms_eval.tasks._task_utils.reasoning_utils import (
    compute_score,
    make_reasoning_doc_to_messages,
)

SYSTEM_PROMPT = (
    "You are a helpful assistant. When the user asks a question, your response must include two parts: "
    "first, the reasoning process enclosed in <think>...</think> tags, then the final answer enclosed in <answer>...</answer> tags."
    "Please provide a clear, concise response within <answer> </answer> tags that directly addresses the question."
)


def ai2d_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question, choices = doc["question"], doc["options"]
    len_choices = len(choices)
    options = [chr(ord("A") + i) for i in range(len_choices)]
    choices_str = "\n".join([f"{option}. {choice}" for option, choice in zip(options, choices)])
    return f"{question}\n{choices_str}"


def ai2d_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


ai2d_doc_to_messages = make_reasoning_doc_to_messages(ai2d_doc_to_visual, ai2d_doc_to_text, system_prompt=SYSTEM_PROMPT)


def ai2d_process_results(doc, results):
    acc_score = 0
    format_score = 0
    question = ai2d_doc_to_text(doc, None)
    len_choices = len(doc["options"])
    options = [chr(ord("A") + i) for i in range(len_choices)]
    ground_truth = options[int(doc["answer"])]
    extra_info = {"question": question}
    for pred in results:
        score_dict = compute_score(data_source="ai2d", solution_str=pred.strip(), ground_truth=ground_truth, extra_info=extra_info)
        acc_score += score_dict["acc_score"]
        format_score += score_dict.get("format_reward_score", 0.0)

    return {"acc_score": acc_score / len(results) if results else 0.0, "format_score": format_score / len(results) if results else 0.0}
