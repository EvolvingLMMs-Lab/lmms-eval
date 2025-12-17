import json
import logging
import os
import random
import re
from collections import defaultdict

import numpy as np
from PIL import Image

eval_logger = logging.getLogger("lmms-eval")

MULTI_CHOICE_DIRECT_PROMPT = "Answer the question with the option's letter from the given choices directly."
COT_PROMPT = "Solve the multiple-choice question and then answer with the option letter from the given choices. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options. Think step by step before answering."
PROMPTS = {"MULTI_CHOICE_DIRECT_PROMPT": MULTI_CHOICE_DIRECT_PROMPT, "COT_PROMPT": COT_PROMPT}


def VisualPuzzles_doc_to_visual(doc):
    return [doc["image"]]


def VisualPuzzles_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = "Question: " + doc["question"].strip()
    options = doc["options"]
    if options != None:
        question += "\nOptions:\n(A) " + options[0] + "\n(B) " + options[1] + "\n(C) " + options[2] + "\n(D) " + options[3]
    else:
        question += "\nOptions: Choose from (A) (B) (C) (D) in the image."
    question += "\n" + PROMPTS[lmms_eval_specific_kwargs["prompt"]]
    return question


def parse_response(response, all_choices, index2ans):
    """
    Return the last letter appearing after 'ANSWER:' in the input text.
    If there's no match, return None.
    """
    pattern = r"Answer:\s*\(([A-Za-z])\)"  # Answer: (A)
    matches = re.findall(pattern, response)
    if matches:
        for match in matches[::-1]:
            if match in all_choices or match.upper() in all_choices:
                return match
    pattern = r"(?<!Final )Answer:\s*([A-Za-z])"  # Answer: A
    matches = re.findall(pattern, response)
    if matches:
        for match in matches[::-1]:
            if match in all_choices or match.upper() in all_choices:
                return match
    pattern = r"Answer:\s*([A-Za-z])"  # Answer: A
    matches = re.findall(pattern, response)
    if matches:
        for match in matches[::-1]:
            if match in all_choices or match.upper() in all_choices:
                return match
    pattern = r"\s*\(([A-Za-z])\)"  # e.g., (A) (B) (C) (D)
    matches = re.findall(pattern, response)
    if matches:
        for match in matches[::-1]:
            if match in all_choices or match.upper() in all_choices:
                return match
    response = " " + response.strip()
    pattern = r"\s*([A-Za-z])\)"  # e.g., A) B) C) D)
    matches = re.findall(pattern, response)
    if matches:
        for match in matches[::-1]:
            if match in all_choices or match.upper() in all_choices:
                return match
    pattern = r"\s*\{([A-Za-z])\}"  # e.g., {A} {B} {C} {D}
    matches = re.findall(pattern, response)
    if matches:
        for match in matches[::-1]:
            if match in all_choices or match.upper() in all_choices:
                return match
    pattern = r"\s*\$([A-Za-z])\$"  # e.g., $A$, $B$, $C$, $D$
    matches = re.findall(pattern, response)
    if matches:
        for match in matches[::-1]:
            if match in all_choices or match.upper() in all_choices:
                return match
    pattern = r" ([A-Da-d])\."  # e.g., A. B. C. D.
    matches = re.findall(pattern, response)
    if matches:
        for match in matches[::-1]:
            if match in all_choices or match.upper() in all_choices:
                return match
    pattern = r" ([A-Da-d])"  # e.g., A B C D
    matches = re.findall(pattern, response)
    if matches and len(response) <= 5:
        for match in matches[::-1]:
            if match in all_choices or match.upper() in all_choices:
                return match
    if index2ans != None:
        for index in all_choices:
            ans = index2ans[index]
            if f"answer: {ans}" in response.lower():
                return index
            if f"answer:{ans}" in response.lower():
                return index
        last_found = None
        last_index = -1
        for index in all_choices:
            ans = index2ans[index]
            idx = response.rfind(ans)
            if idx > last_index:
                last_found = index
                last_index = idx
        if last_found:
            return last_found
    return random.choice(all_choices)


def VisualPuzzles_process_result(doc, results):
    """
    Process results with category-based metrics

    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (category), value: metric dict
    """
    pred = results[0].strip()
    all_choices = ["A", "B", "C", "D"]
    if doc["options"] == None:
        index2ans = None
    else:
        index2ans = {all_choices[i]: doc["options"][i] for i in range(4)}
    pred = parse_response(pred, all_choices, index2ans)
    target = doc["answer"]

    # Calculate score
    score = 1.0 if pred.lower() == target.lower() else 0.0

    # Get category from doc
    category = doc.get("category", "Unknown")

    # Return results for both category-specific and overall metrics
    result_dict = {
        category: {
            "question_id": doc.get("id", doc.get("idx", "unknown")),
            "category": category,
            "score": score
        },
        "average": {
            "question_id": doc.get("id", doc.get("idx", "unknown")),
            "category": category,
            "score": score
        }
    }

    return result_dict


def VisualPuzzles_aggregate_results(results):
    """
    Aggregate results by category

    Args:
        results: a list of values returned by process_results
    Returns:
        Average score for the category
    """
    category_scores = defaultdict(list)

    for result in results:
        score = result["score"]
        category = result["category"]
        category_scores[category].append(score)

    # Calculate average for each category
    category_avg_scores = {}
    for category, scores in category_scores.items():
        avg_score = sum(scores) / len(scores) if scores else 0.0
        category_avg_scores[category] = avg_score
        eval_logger.info(f"{category}: {avg_score:.4f} ({len(scores)} samples)")

    # Calculate overall average
    all_scores = [score for scores in category_scores.values() for score in scores]
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

    return avg_score
