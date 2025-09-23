import random
from typing import Any, Dict, List

import numpy as np


def medqa_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Dict[str, Any]):
    """
    Build the MCQ prompt from MEDQA sample.

    Expected doc fields (from `lmms-lab/MEDQA` parquet):
    - "question": str
    - "options": dict mapping letters to option strings (e.g., {"A": "...", "B": "..."})
    - Some samples may also expose choices as list-like; we normalize to a lettered block.
    - We do not use visuals for MEDQA.
    """
    question = doc.get("question", "").strip()

    # Normalize options into A..E style lines
    options = doc.get("options")
    if isinstance(options, dict):
        # Keep only A-E in sorted letter order if present
        ordered_keys = [k for k in ["A", "B", "C", "D", "E"] if k in options]
        options_block = "\n".join([f"{k}. {str(options[k]).strip()}" for k in ordered_keys])
    elif isinstance(options, list):
        letters = ["A", "B", "C", "D", "E"]
        options_block = "\n".join([f"{letters[i]}. {str(opt).strip()}" for i, opt in enumerate(options)])
    else:
        # Fallback: try to format if already string-like
        options_block = str(options) if options is not None else ""

    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    prompt = f"{question}\n{options_block}"
    return f"{pre_prompt}{prompt}{post_prompt}"


def medqa_doc_to_target(doc: Dict[str, Any]):
    """
    Return the ground-truth answer letter.

    MEDQA on HF commonly provides either:
    - "answer_idx": a letter like "A"/"B"/... OR
    - "answer": a full string like "C" or the option text. We prioritize letter if available.
    """
    # Prefer explicit answer letter field when present
    if "answer_idx" in doc and isinstance(doc["answer_idx"], str) and len(doc["answer_idx"]) == 1:
        return doc["answer_idx"].strip()

    # Some variants store the letter in "answer" directly
    ans = doc.get("answer")
    if isinstance(ans, str) and len(ans.strip()) == 1 and ans.strip().upper() in ["A", "B", "C", "D", "E"]:
        return ans.strip().upper()

    # If answer is provided as text, try to map back to a letter via options
    options = doc.get("options")
    if isinstance(options, dict) and isinstance(ans, str):
        for k, v in options.items():
            if isinstance(v, str) and v.strip() == ans.strip():
                return k

    # Fallback: unknown -> choose a dummy; evaluation will mark as incorrect
    return "A"


def medqa_doc_to_choice(doc: Dict[str, Any]) -> List[str]:
    # Detect how many choices are present and return corresponding letters
    if isinstance(doc.get("options"), dict):
        present = [k for k in ["A", "B", "C", "D", "E"] if k in doc["options"]]
        if present:
            return present
    if isinstance(doc.get("options"), list):
        n = min(len(doc["options"]), 5)
        return ["A", "B", "C", "D", "E"][:n]
    # Default to 5-way if uncertain
    return ["A", "B", "C", "D", "E"]


def medqa_process_results(doc: Dict[str, Any], result: List[str]):
    """
    Parse model output and compute accuracy against the gold letter.
    We robustly extract a single letter from the response.
    """
    response = result[0].strip()
    all_choices = medqa_doc_to_choice(doc)
    pred = _parse_multi_choice_response(response, all_choices)
    gt_ans = medqa_doc_to_target(doc)
    score = 1.0 if pred == gt_ans else 0.0
    return {"accuracy": score}


def _parse_multi_choice_response(response: str, all_choices: List[str]) -> str:
    # Clean punctuation around the response
    for ch in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(ch)
    response = " " + response + " "

    candidates = []
    # (A) style
    for c in all_choices:
        if f"({c})" in response:
            candidates.append(c)

    # plain letter surrounded by spaces
    if len(candidates) == 0:
        for c in all_choices:
            if f" {c} " in response:
                candidates.append(c)

    # A., B., etc.
    if len(candidates) == 0:
        for c in all_choices:
            if f"{c}." in response:
                candidates.append(c)

    if len(candidates) == 0:
        return random.choice(all_choices)
    if len(candidates) > 1:
        # choose the last occurrence to mitigate explanations mentioning multiple letters
        start_indexes = [response.rfind(f" {can} ") for can in candidates]
        return candidates[int(np.argmax(start_indexes))]
    return candidates[0]
