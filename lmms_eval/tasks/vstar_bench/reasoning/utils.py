from lmms_eval.tasks._task_utils.reasoning_utils import compute_score

SYSTEM_PROMPT = (
    "You are a helpful assistant. When user asks a question, your response must include two parts: "
    "first, reasoning process enclosed in <analysis>...</analysis> tags, then final answer enclosed in <answer>...</answer> tags."
    "Please provide a clear, concise response within <answer> </answer> tags that directly addresses to question."
)


def vstar_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def vstar_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["text"]

    import re
    options_match = re.findall(r"\([A-D]\)\s*([^()]+?)(?=\s*\([A-D]\)|$)", question)

    if options_match:
        question_text = question.split("(A)")[0].strip()
        option_letters = ["A", "B", "C", "D"]
        options_formatted = []
        for i, option in enumerate(options_match):
            if i < len(option_letters):
                options_formatted.append(f"{option_letters[i]}. {option.strip()}")
        question = f"{question_text}\n" + "\n".join(options_formatted)

    if lmms_eval_specific_kwargs:
        if "pre_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["pre_prompt"]:
            question = f"{lmms_eval_specific_kwargs['pre_prompt']}{question}"
        if "post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["post_prompt"]:
            question = f"{question}{lmms_eval_specific_kwargs['post_prompt']}"

    return question


def vstar_reasoning_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    question = vstar_doc_to_text(doc, lmms_eval_specific_kwargs)
    visuals = vstar_doc_to_visual(doc)
    system_messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
    messages = [{"role": "user", "content": []}]
    for visual in visuals:
        messages[0]["content"].append({"type": "image", "url": visual})
    messages[0]["content"].append({"type": "text", "text": question.strip()})
    messages = system_messages + messages
    return messages


def vstar_reasoning_process_results(doc, results):
    question = vstar_doc_to_text(doc, None)
    ground_truth = doc["label"].strip().upper()
    category = doc.get("category", "unknown")
    extra_info = {"question": question, "category": category}

    acc_score = 0
    format_score = 0
    for pred in results:
        score_dict = compute_score(data_source="vstar_bench", solution_str=pred.strip(), ground_truth=ground_truth, extra_info=extra_info)
        acc_score += score_dict["acc_score"]
        format_score += score_dict.get("format_reward_score", 0.0)

    acc_score = acc_score / len(results) if results else 0.0
    format_score = format_score / len(results) if results else 0.0

    return {
        f"vstar_{category}_acc_score": acc_score,
        f"vstar_{category}_format_score": format_score,
        "vstar_overall_acc_score": acc_score,
        "vstar_overall_format_score": format_score,
    }
