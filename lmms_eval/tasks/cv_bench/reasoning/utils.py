from lmms_eval.tasks._task_utils.reasoning_utils import compute_score

SYSTEM_PROMPT = (
    "You are a helpful assistant. When user asks a question, your response must include two parts: "
    "first, reasoning process enclosed in <analysis>...</analysis> tags, then final answer enclosed in <answer>...</answer> tags."
    "Please provide a clear, concise response within <answer> </answer> tags that directly addresses to question."
)


def cv_bench_doc_to_visual(doc: dict) -> list:
    return [doc["image"].convert("RGB")]


def cv_bench_doc_to_text(doc: dict, lmms_eval_specific_kwargs=None) -> str:
    num_choices = len(doc["choices"])
    choice_letters = ", ".join([chr(65 + i) for i in range(num_choices)])
    prompt = lmms_eval_specific_kwargs.get("pre_prompt", "").format(choice_letters) if lmms_eval_specific_kwargs else ""
    prompt += "\n" + doc["prompt"]
    return prompt


def cv_bench_reasoning_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    question = cv_bench_doc_to_text(doc, lmms_eval_specific_kwargs)
    visuals = cv_bench_doc_to_visual(doc)
    system_messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
    messages = [{"role": "user", "content": []}]
    for visual in visuals:
        messages[0]["content"].append({"type": "image", "url": visual})
    messages[0]["content"].append({"type": "text", "text": question.strip()})
    messages = system_messages + messages
    return messages


def cv_bench_reasoning_process_results(doc, results):
    acc_score = 0
    format_score = 0
    question = cv_bench_doc_to_text(doc, None)
    ground_truth = doc["answer"]
    extra_info = {"question": question}
    for pred in results:
        score_dict = compute_score(data_source="cv_bench", solution_str=pred.strip(), ground_truth=ground_truth, extra_info=extra_info)
        acc_score += score_dict["acc_score"]
        format_score += score_dict.get("format_reward_score", 0.0)

    return {"cv_bench_acc_score": acc_score / len(results) if results else 0.0, "cv_bench_format_score": format_score / len(results) if results else 0.0}
