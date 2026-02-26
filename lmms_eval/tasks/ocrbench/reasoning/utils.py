from lmms_eval.tasks._task_utils.reasoning_utils import compute_score, extract_anwser_tag, format_reward
from lmms_eval.tasks.ocrbench.utils import ocrbench_process_results

SYSTEM_PROMPT = (
    "You are a helpful assistant. When user asks a question, your response must include two parts: "
    "first, reasoning process enclosed in <analysis>...</analysis> tags, then final answer enclosed in <answer>...</answer> tags."
    "Please provide a clear, concise response within <answer> </answer> tags that directly addresses to question."
)


def ocrbench_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def ocrbench_doc_to_text(doc, lmms_eval_specific_kwargs):
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") if lmms_eval_specific_kwargs else ""
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "") if lmms_eval_specific_kwargs else ""
    question = doc["question"].strip()
    return f"{pre_prompt}{question}{post_prompt}"


def ocrbench_reasoning_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    question = ocrbench_doc_to_text(doc, lmms_eval_specific_kwargs)
    visuals = ocrbench_doc_to_visual(doc)
    system_messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
    messages = [{"role": "user", "content": []}]
    for visual in visuals:
        messages[0]["content"].append({"type": "image", "url": visual})
    messages[0]["content"].append({"type": "text", "text": question.strip()})
    messages = system_messages + messages
    return messages


def ocrbench_reasoning_process_results(doc, results):
    question = ocrbench_doc_to_text(doc, None)
    ground_truth = doc["answer"]
    extra_info = {"question": question, "dataset": doc["dataset"]}

    acc_score = 0
    format_score = 0
    for pred in results:
        format_score += format_reward(pred)
        pred = extract_anwser_tag(pred).strip()
        score_dict = ocrbench_process_results(doc, [pred])
        acc_score += score_dict["ocrbench_accuracy"]["score"]

    return {"ocrbench_accuracy_acc_score": acc_score / len(results) if results else 0.0, "ocrbench_accuracy_format_score": format_score / len(results) if results else 0.0}
