from lmms_eval.tasks._task_utils.reasoning_utils import (
    compute_score,
    make_reasoning_doc_to_messages,
)

SYSTEM_PROMPT = (
    "You are a helpful assistant. When the user asks a question, your response must include two parts: "
    "first, the reasoning process enclosed in <think>...</think> tags, then the final answer enclosed in <answer>...</answer> tags."
    "Please provide a clear, concise response within <answer> </answer> tags that directly addresses the question."
)


def realworldqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"].strip()
    return question


def realworldqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


realworldqa_doc_to_messages = make_reasoning_doc_to_messages(realworldqa_doc_to_visual, realworldqa_doc_to_text, system_prompt=SYSTEM_PROMPT)


def realworldqa_process_results(doc, results):
    acc_score = 0
    format_score = 0
    question = realworldqa_doc_to_text(doc, None)
    ground_truth = doc["answer"].lower().strip()
    extra_info = {"question": question}
    for pred in results:
        score_dict = compute_score(data_source="realworldqa", solution_str=pred.strip(), ground_truth=ground_truth, extra_info=extra_info)
        acc_score += score_dict["acc_score"]
        format_score += score_dict.get("format_reward_score", 0.0)

    return {"acc_score": acc_score / len(results) if results else 0.0, "format_score": format_score / len(results) if results else 0.0}
