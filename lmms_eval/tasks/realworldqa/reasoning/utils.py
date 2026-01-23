from lmms_eval.tasks._task_utils.reasoning_utils import compute_score

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


def realworldqa_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    question = realworldqa_doc_to_text(doc, lmms_eval_specific_kwargs)
    visuals = realworldqa_doc_to_visual(doc)
    system_messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
    messages = [{"role": "user", "content": []}]
    messages[0]["content"].append({"type": "image", "url": visuals[0]})
    messages[0]["content"].append({"type": "text", "text": question.strip()})
    messages = system_messages + messages
    return messages


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
