from lmms_eval.tasks._task_utils.reasoning_utils import compute_score

SYSTEM_PROMPT = (
    "You are a helpful assistant. When user asks a question, your response must include two parts: "
    "first, reasoning process enclosed in <analysis>...</analysis> tags, then final answer enclosed in <answer>...</answer> tags."
    "Please provide a clear, concise response within <answer> </answer> tags that directly addresses to question."
)


def seed_doc_to_visual(doc):
    return [image.convert("RGB") for image in doc["image"]]


def seed_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    question += "\n" + f"A. {doc['choice_a']}\n"
    question += f"B. {doc['choice_b']}\n"
    question += f"C. {doc['choice_c']}\n"
    question += f"D. {doc['choice_d']}"
    return f"{question}\nAnswer with the option's letter from the given choices directly."


def seed_reasoning_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    question = seed_doc_to_text(doc, lmms_eval_specific_kwargs)
    visuals = seed_doc_to_visual(doc)
    system_messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
    messages = [{"role": "user", "content": []}]
    for visual in visuals:
        messages[0]["content"].append({"type": "image", "url": visual})
    messages[0]["content"].append({"type": "text", "text": question.strip()})
    messages = system_messages + messages
    return messages


def seed_reasoning_process_results(doc, results):
    data_type = doc["data_type"]
    acc_score = 0
    format_score = 0
    question = seed_doc_to_text(doc, None)
    ground_truth = doc["answer"]
    extra_info = {"question": question}
    for pred in results:
        score_dict = compute_score(data_source="seedbench", solution_str=pred.strip(), ground_truth=ground_truth, extra_info=extra_info)
        acc_score += score_dict["acc_score"]
        format_score += score_dict.get("format_reward_score", 0.0)

    return {
        f"seed_{data_type}_acc_score": acc_score / len(results) if results else 0.0,
        f"seed_{data_type}_format_score": format_score / len(results) if results else 0.0,
        "seed_all_acc_score": acc_score / len(results) if results else 0.0,
        "seed_all_format_score": format_score / len(results) if results else 0.0,
    }
