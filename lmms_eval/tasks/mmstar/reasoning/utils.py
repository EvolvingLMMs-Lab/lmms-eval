import string

from lmms_eval.tasks._task_utils.reasoning_utils import compute_score

SYSTEM_PROMPT = (
    "You are a helpful assistant. When the user asks a question, your response must include two parts: "
    "first, the reasoning process enclosed in <analysis>...</analysis> tags, then the final answer enclosed in <answer>...</answer> tags."
    "Please provide a clear, concise response within <answer> </answer> tags that directly addresses the question."
)


replace_prompt = " Please answer yes or no."


def mmstar_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def mmstar_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    lmms_eval_specific_kwargs = {} if not lmms_eval_specific_kwargs else lmms_eval_specific_kwargs
    question = doc["question"].strip()
    if "pre_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["pre_prompt"] != "":
        question = question.replace(replace_prompt, "")
        question = f"{lmms_eval_specific_kwargs['pre_prompt']}{question}"
    if "post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["post_prompt"] != "":
        question = question.replace(replace_prompt, "")
        question = f"{question}{lmms_eval_specific_kwargs['post_prompt']}"
    return question


def mmstar_reasoning_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    question = mmstar_doc_to_text(doc, lmms_eval_specific_kwargs)
    visuals = mmstar_doc_to_visual(doc)
    system_messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
    messages = [{"role": "user", "content": []}]
    messages[0]["content"].append({"type": "image", "url": visuals[0]})
    messages[0]["content"].append({"type": "text", "text": question.strip()})
    messages = system_messages + messages
    return messages


def mmstar_reasoning_process_results(doc, results):
    acc_score = 0
    format_score = 0
    question = mmstar_doc_to_text(doc, None)
    ground_truth = doc["answer"]
    extra_info = {"question": question}
    for pred in results:
        score_dict = compute_score(data_source="mmstar", solution_str=pred.strip(), ground_truth=ground_truth, extra_info=extra_info)
        acc_score += score_dict["acc_score"]
        format_score += score_dict.get("format_reward_score", 0.0)

    return {"acc_score": acc_score / len(results) if results else 0.0, "format_score": format_score / len(results) if results else 0.0}
