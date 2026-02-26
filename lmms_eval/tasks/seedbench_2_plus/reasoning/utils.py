from lmms_eval.tasks._task_utils.reasoning_utils import compute_score

SYSTEM_PROMPT = (
    "You are a helpful assistant. When user asks a question, your response must include two parts: "
    "first, reasoning process enclosed in <analysis>...</analysis> tags, then final answer enclosed in <answer>...</answer> tags."
    "Please provide a clear, concise response within <answer> </answer> tags that directly addresses to question."
)


def parse_choice_img(choice: str, img_token: str = "<image>"):
    if "jpg" in choice or "png" in choice:
        return img_token
    return choice


def seed_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def seed_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    img_token = lmms_eval_specific_kwargs.get("img_token", "<image>") if lmms_eval_specific_kwargs else "<image>"
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "Answer with the option's letter from the given choices directly.") if lmms_eval_specific_kwargs else "Answer with the option's letter from the given choices directly."

    question.replace("<img>", img_token)
    question += "\n" + f"A. {parse_choice_img(doc['choice_A'], img_token)}\n"
    question += f"B. {parse_choice_img(doc['choice_B'], img_token)}\n"
    question += f"C. {parse_choice_img(doc['choice_C'], img_token)}\n"
    question += f"D. {parse_choice_img(doc['choice_D'], img_token)}"

    return f"{question}\n{post_prompt}"


def seed_reasoning_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    question = seed_doc_to_text(doc, lmms_eval_specific_kwargs)
    visuals = seed_doc_to_visual(doc)
    system_messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
    messages = [{"role": "user", "content": []}]
    messages[0]["content"].append({"type": "image", "url": visuals[0]})
    messages[0]["content"].append({"type": "text", "text": question.strip()})
    messages = system_messages + messages
    return messages


def seed_reasoning_process_results(doc, results):
    data_type = doc["question_image_type"].capitalize()
    acc_score = 0
    format_score = 0
    question = seed_doc_to_text(doc, None)
    ground_truth = doc["answer"]
    extra_info = {"question": question}
    for pred in results:
        score_dict = compute_score(data_source="seedbench_2_plus", solution_str=pred.strip(), ground_truth=ground_truth, extra_info=extra_info)
        acc_score += score_dict["acc_score"]
        format_score += score_dict.get("format_reward_score", 0.0)

    return {
        f"seedbench_2_plus_{data_type}_acc_score": acc_score / len(results) if results else 0.0,
        f"seedbench_2_plus_{data_type}_format_score": format_score / len(results) if results else 0.0,
        "seedbench_2_plus_all_acc_score": acc_score / len(results) if results else 0.0,
        "seedbench_2_plus_all_format_score": format_score / len(results) if results else 0.0,
    }
