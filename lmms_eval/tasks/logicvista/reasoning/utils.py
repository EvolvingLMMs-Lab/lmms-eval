from lmms_eval.tasks._task_utils.reasoning_utils import (
    make_reasoning_doc_to_messages,
    make_reasoning_process_results,
)

SYSTEM_PROMPT = (
    "You are a helpful assistant. When the user asks a question, your response must include two parts: "
    "first, the reasoning process enclosed in <think>...</think> tags, then the final answer enclosed in <answer>...</answer> tags."
    "Please provide a clear, concise response within <answer> </answer> tags that directly addresses the question."
)


def logicvista_doc_to_text_cot(doc, lmms_eval_specific_kwargs=None):
    return doc["question"]


def logicvista_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


logicvista_doc_to_messages_cot = make_reasoning_doc_to_messages(logicvista_doc_to_visual, logicvista_doc_to_text_cot, system_prompt=SYSTEM_PROMPT)


logicvista_reasoning_process_results = make_reasoning_process_results("logicvista", logicvista_doc_to_text_cot)
