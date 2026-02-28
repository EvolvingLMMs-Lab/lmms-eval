from lmms_eval.tasks._task_utils.reasoning_utils import (
    make_reasoning_doc_to_messages,
    make_reasoning_process_results,
)
from lmms_eval.tasks.phyx.utils import decode_base64_to_image

SYSTEM_PROMPT = (
    "You are a helpful assistant. When the user asks a question, your response must include two parts: "
    "first, the reasoning process enclosed in <think>...</think> tags, then the final answer enclosed in <answer>...</answer> tags."
    "Please provide a clear, concise response within <answer> </answer> tags that directly addresses the question."
)


def phyx_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    query_prompt = doc["question"]
    return query_prompt


def phyx_doc_to_visual(doc):
    image = decode_base64_to_image(doc["image"])
    return [image]


phyx_doc_to_messages = make_reasoning_doc_to_messages(phyx_doc_to_visual, phyx_doc_to_text, system_prompt=SYSTEM_PROMPT)


phyx_process_results_mc = make_reasoning_process_results("phyx_mc", phyx_doc_to_text)


phyx_process_results = make_reasoning_process_results("phyx_oe", phyx_doc_to_text)
