from lmms_eval.tasks._task_utils.reasoning_utils import format_reward, extract_anwser_tag
from lmms_eval.api.metrics import anls

SYSTEM_PROMPT = (
    "You are a helpful assistant. When user asks a question, your response must include two parts: "
    "first, reasoning process enclosed in <analysis>...</analysis> tags, then final answer enclosed in <answer>...</answer> tags."
    "Please provide a clear, concise response within <answer> </answer> tags that directly addresses to question."
)


def infovqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def infovqa_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") if lmms_eval_specific_kwargs else ""
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "") if lmms_eval_specific_kwargs else ""
    return f"{pre_prompt}{question}{post_prompt}"


def infovqa_reasoning_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    question = infovqa_doc_to_text(doc, lmms_eval_specific_kwargs)
    visuals = infovqa_doc_to_visual(doc)
    system_messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
    messages = [{"role": "user", "content": []}]
    for visual in visuals:
        messages[0]["content"].append({"type": "image", "url": visual})
    messages[0]["content"].append({"type": "text", "text": question.strip()})
    messages = system_messages + messages
    return messages


def infovqa_reasoning_process_results(doc, results):
    question = infovqa_doc_to_text(doc, None)
    ground_truth = doc["answers"]
    extra_info = {"question": question}

    acc_score = 0
    format_score = 0
    for pred in results:
        pred = extract_anwser_tag(pred).strip()
        anls_score = anls(ground_truth, results)
        acc_score += anls_score['anls']
        format_score += format_reward(pred)

    return {"anls_acc_score": acc_score / len(results) if results else 0.0, "anls_format_score": format_score / len(results) if results else 0.0}
