import os
import re
import signal
from collections import Counter
from typing import Dict, List, Optional

import datasets

from lmms_eval.tasks._task_utils.reasoning_utils import compute_score

SYSTEM_PROMPT = (
    "You are a helpful assistant. When the user asks a question, your response must include two parts: "
    "first, the reasoning process enclosed in <think>...</think> tags, then the final answer enclosed in <answer>...</answer> tags."
    "Please provide a clear, concise response within <answer> </answer> tags that directly addresses the question."
)


def doc_to_text(doc: dict) -> str:
    return doc.get("problem", doc.get("question"))


def doc_to_messages(doc: dict):
    question = doc_to_text(doc)
    system_messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
    messages = [{"role": "user", "content": []}]
    messages[0]["content"].append({"type": "text", "text": question.strip()})
    messages = system_messages + messages
    return messages


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        solution = doc.get("solution", doc.get("orig_solution", doc.get("orig_orig_solution")))
        problem = doc.get("problem", doc.get("question"))
        answer = doc.get("answer", doc.get("orig_answer", doc.get("orig_orig_answer")))
        if solution is None:
            print("Warning: No solution found; DOC:", doc)
        out_doc = {
            "problem": problem,
            "solution": solution,
            "answer": answer,
        }
        if getattr(doc, "few_shot", None) is not None:
            out_doc["few_shot"] = True
        return out_doc

    return dataset.map(_process_doc)


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    metrics = {"exact_match": None, "extracted_answers": []}
    # bp()
    # Multiple results -> we are measuring cov/maj etc
    if len(results) > 1:
        n_res = len(results)  # e.g. 64
        n_res_list = [2**i for i in range(1, int(n_res.bit_length()))]  # e.g. [2, 4, 8, 16, 32, 64]
        metrics = {
            **metrics,
            "exact_matches": [],
            **{f"cov@{n}": -1 for n in n_res_list},
            **{f"maj@{n}": -1 for n in n_res_list},
            **{f"avg@{n}": -1 for n in n_res_list},
        }
    else:
        n_res_list = []
        metrics["exact_matches"] = []

    if isinstance(doc["answer"], str) and doc["answer"].isdigit():
        gt = str(int(doc["answer"]))  # 023 -> 23
    else:
        gt = str(doc["answer"])

    question = doc_to_text(doc)
    extra_info = {"question": question}
    for i, a in enumerate(results, start=1):

        score_dict = compute_score(data_source="aime", solution_str=a.strip(), ground_truth=gt, extra_info=extra_info)
        acc_score = score_dict["acc_score"]

        metrics["extracted_answers"].append(score_dict["predict_str"])
        if i == 1:
            metrics["exact_match"] = acc_score
            if "exact_matches" in metrics:
                metrics["exact_matches"].append(acc_score)
        elif i > 1:
            metrics["exact_matches"].append(acc_score)
            if i in n_res_list:
                metrics[f"cov@{i}"] = int(1 in metrics["exact_matches"])
                metrics[f"avg@{i}"] = sum(metrics["exact_matches"]) / i
                metrics[f"maj@{i}"] = int((sum(metrics["exact_matches"]) / i) > 0.5)

    return metrics


def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]
