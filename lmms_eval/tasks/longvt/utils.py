import os
import re
from pathlib import Path

import yaml

from lmms_eval.tasks._task_utils.reasoning_utils import compute_score

hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "longvt_reasoning.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]

SYSTEM_PROMPT = (
    "You are a helpful assistant. When the user asks a question, your response must include two parts: "
    "first, the reasoning process enclosed in <think>...</think> tags, then the final answer enclosed in <answer>...</answer> tags."
    "Please provide a clear, concise response within <answer> </answer> tags that directly addresses the question."
)

TOOL_PROMPT = "Think first, call **crop_video** if needed, then answer. Format strictly as:  <think>...</think>  " "<tool_call>...</tool_call> (if tools needed)  <answer>...</answer>."


def longvt_doc_to_visual(doc):
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = doc["video_path"]
    assert os.path.exists(os.path.join(cache_dir, video_path))
    video_path = os.path.join(cache_dir, video_path)
    return [video_path]


def longvt_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is not None:
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
        if post_prompt:
            return doc["question"] + " " + post_prompt
        return doc["question"]
    else:
        return doc["question"]


def longvt_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    question = longvt_doc_to_text(doc, lmms_eval_specific_kwargs)
    visuals = longvt_doc_to_visual(doc)
    system_messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
    messages = [{"role": "user", "content": []}]
    messages[0]["content"].append({"type": "video", "url": visuals[0]})
    messages[0]["content"].append({"type": "text", "text": question.strip()})
    messages = system_messages + messages
    return messages


def longvt_tool_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    question = longvt_doc_to_text(doc, lmms_eval_specific_kwargs)
    visuals = longvt_doc_to_visual(doc)
    messages = [{"role": "user", "content": []}]
    messages[0]["content"].append({"type": "video", "url": visuals[0]})
    messages[0]["content"].append({"type": "text", "text": question.strip() + " " + TOOL_PROMPT})
    return messages


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is" "The correct option is",
        "Best answer:" "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""

    matches = re.search(r"[ABCD]", s)
    if matches is None:
        return ""
    return matches[0]


def longvt_process_results(doc, results):
    acc_score = 0
    format_score = 0
    question = longvt_doc_to_text(doc, lmms_eval_specific_kwargs=None)
    extra_info = {"question": question}
    answer = doc["answer"]
    for pred in results:
        score_dict = compute_score(data_source="longvt", solution_str=pred.strip(), ground_truth=answer, extra_info=extra_info)
        acc_score += score_dict["acc_score"]
        format_score += score_dict.get("format_reward_score", 0.0)

    return {"acc_score": acc_score / len(results) if results else 0.0, "format_score": format_score / len(results) if results else 0.0}
