import re

from lmms_eval.tasks.ocrbench_v2.utils import (
    ocrbench_v2_aggregate_accuracy,
    ocrbench_v2_doc_to_target,
    ocrbench_v2_doc_to_text,
    ocrbench_v2_doc_to_visual,
)

SYSTEM_PROMPT = (
    "You are a helpful assistant. When the user asks a question, your response must include two parts: "
    "first, the reasoning process enclosed in <think>...</think> tags, then the final answer enclosed in <answer>...</answer> tags."
    "Please provide a clear, concise response within <answer> </answer> tags that directly addresses the question."
)


def extract_answer_tag(predict_str):
    """Extract the answer tag from the prediction string.

    Args:
        predict_str (str): The prediction string containing the answer tag.

    Returns:
        str: The extracted answer from <answer> tags, or the original string if not found.
    """
    pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    match_result = re.search(pattern, predict_str)
    if match_result:
        return match_result.group(1).strip()

    return predict_str.strip()


def ocrbench_v2_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    question = ocrbench_v2_doc_to_text(doc)
    visuals = ocrbench_v2_doc_to_visual(doc)
    system_messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
    messages = [{"role": "user", "content": []}]
    messages[0]["content"].append({"type": "image", "url": visuals[0]})
    messages[0]["content"].append({"type": "text", "text": question.strip()})
    messages = system_messages + messages
    return messages


def ocrbench_v2_process_results_reasoning(doc, results):
    from lmms_eval.tasks.ocrbench_v2.utils import ocrbench_v2_process_results

    extracted_results = []
    for pred in results:
        extracted_answer = extract_answer_tag(pred)
        extracted_results.append(extracted_answer)

    return ocrbench_v2_process_results(doc, extracted_results)
