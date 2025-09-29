import random
import re
from typing import Dict, List, Tuple

import numpy as np

assertion_prompt = """Answer the following multiple choice question. There is only one correct answer. The last line of your response should be in the format 'Answer: $LETTER' (without quotes), where LETTER is one of A or B."""

mcq_prompt = """Answer the following multiple choice question. There is only one correct answer. The last line of your response should be in the format 'Answer: $LETTER' (without quotes), where LETTER is one of A, B, C, or D."""

def csbench_mcq_doc_to_text(doc: Dict, lmms_eval_specific_kwargs: Dict) -> str:
    q = doc["Question"]
    a = doc["A"]
    b = doc["B"]
    c = doc["C"]
    d = doc["D"]
    question = f"{assertion_prompt}\nQuestion: {q}\nA: {a}\nB: {b}\nC: {c}\nD: {d}\n"
    return question

def csbench_assertion_doc_to_text(doc: Dict, lmms_eval_specific_kwargs: Dict) -> str:
    q = doc["Question"]
    question = f"{assertion_prompt}\nQuestion: {q}\n A: True\n B: False\n"
    return question

def csbench_doc_to_target(doc: Dict) -> str:
    if doc["Format"].strip() == "Multiple-choice":
        return doc["Answer"].strip().upper()
    else:
        return "A" if doc["Answer"].strip() == "True" else "B"

def csbench_doc_to_choice(doc: Dict) -> List[str]:
    if doc["Format"].strip() == "Multiple-choice":
        return ["A", "B", "C", "D"]
    else:
        return ["A", "B"]

def parse_multi_choice_response(response, all_choices):
    """
    Parse the prediction from the generated response.
    Return the predicted choice letter e.g., A, B, C, D.
    """
    # Clean response of unwanted characters
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # Add space to avoid partial match

    candidates = []
    # Look for choices with parentheses, e.g., (A)
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)

    # Look for simple choices, e.g., A, B, C
    if len(candidates) == 0:
        for choice in all_choices:
            if f" {choice} " in response:
                candidates.append(choice)

    # Look for choices with periods, e.g., A., B., C.
    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice}." in response:
                candidates.append(choice)

    # If no candidates, randomly choose one
    if len(candidates) == 0:
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        # If more than one candidate, choose the last one found
        start_indexes = [response.rfind(f" {can} ") for can in candidates]
        pred_index = candidates[np.argmax(start_indexes)]
    else:
        # If only one candidate, use it
        pred_index = candidates[0]

    return pred_index


def csbench_process_results(doc: Dict, result: List[str]) -> Dict[str, float]:
    pred = parse_multi_choice_response(result[0], csbench_doc_to_choice(doc))
    gt = csbench_doc_to_target(doc)
    score = 1.0 if pred == gt else 0.0
    return {"accuracy": score}
