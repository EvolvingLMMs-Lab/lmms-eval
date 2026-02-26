from lmms_eval.tasks._task_utils.reasoning_utils import compute_score

SYSTEM_PROMPT = (
    "You are a helpful assistant. When user asks a question, your response must include two parts: "
    "first, reasoning process enclosed in <analysis>...</analysis> tags, then final answer enclosed in <answer>...</answer> tags."
    "Please provide a clear, concise response within <answer> </answer> tags that directly addresses to question."
)


def countbenchqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def countbenchqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    if doc.get("text"):
        question = f"{doc['text']}\n\n{question}"
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") if lmms_eval_specific_kwargs else ""
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "") if lmms_eval_specific_kwargs else ""
    return f"{pre_prompt}{question}{post_prompt}"


def countbenchqa_reasoning_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    question = countbenchqa_doc_to_text(doc, lmms_eval_specific_kwargs)
    visuals = countbenchqa_doc_to_visual(doc)
    system_messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
    messages = [{"role": "user", "content": []}]
    for visual in visuals:
        messages[0]["content"].append({"type": "image", "url": visual})
    messages[0]["content"].append({"type": "text", "text": question.strip()})
    messages = system_messages + messages
    return messages


def countbenchqa_reasoning_process_results(doc, results):
    question = countbenchqa_doc_to_text(doc, None)
    ground_truth = str(doc["number"])
    extra_info = {"question": question}

    acc_score = 0
    format_score = 0
    for pred in results:
        score_dict = compute_score(data_source="countbenchqa", solution_str=pred.strip(), ground_truth=ground_truth, extra_info=extra_info)
        acc_score += score_dict["acc_score"]
        format_score += score_dict.get("format_reward_score", 0.0)

    return {"acc_score": acc_score / len(results) if results else 0.0, "format_score": format_score / len(results) if results else 0.0}
