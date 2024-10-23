import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

eval_logger = logging.getLogger("lmms-eval")
hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)


def cinepile_doc_to_visual(doc):
    with open(Path(__file__).parent / "cinepile.yaml", "r") as f:
        raw_data = f.readlines()
        safe_data = []
        for i, line in enumerate(raw_data):
            # remove function definition since yaml load cannot handle it
            if "!function" not in line:
                safe_data.append(line)
    task_name = yaml.safe_load("".join(safe_data))["task"]
    cache_dir = os.path.join(base_cache_dir, task_name)

    video_path = doc["videoID"] + ".mp4"
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    elif os.path.exists(video_path.replace("mp4", "MP4")):
        video_path = video_path.replace("mp4", "MP4")
    elif os.path.exists(video_path.replace("mp4", "mkv")):
        video_path = video_path.replace("mp4", "mkv")
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


def format_question_and_options(question, options):
    """
    Formats a question and a list of options into a single string with options labeled A, B, C, etc.

    Parameters:
    - question (str): The question to be formatted.
    - options (list of str): The options for the question.

    Returns:
    - str: The formatted question and options.
    """
    formatted_string = f"{question}\n"
    option_labels = [chr(ord("A") + i) for i in range(len(options))]  # Generate option labels dynamically

    for label, option in zip(option_labels, options):
        formatted_string += f"- {label}) {option}\n"

    return formatted_string


def cinepile_doc_to_text(doc, model_specific_prompt_kwargs=None):
    formatted_question = format_question_and_options(doc["question"], doc["choices"])
    model_input = f"{model_specific_prompt_kwargs['pre_prompt']}\n\n**Subtitles:**\n{doc['subtitles']}\n\n{formatted_question}\n{model_specific_prompt_kwargs['post_prompt']}"
    return model_input


def cinepile_doc_to_target(doc):
    ans_key_to_option = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
    answer_key, answer_key_position = doc["answer_key"], doc["answer_key_position"]

    answer = f"{ans_key_to_option[answer_key_position]}) {answer_key}"
    return answer


def normalize_string(input_string):
    """
    Extracts and returns the option number and option text from a given string.
    The option number is expected to be a single letter followed by an optional bracket and/or period.
    The option text is any text following the option number and its bracket/period.
    If the string does not contain an option number, the entire string is considered as the option text.
    """
    input_string = input_string.replace("*", "").strip()
    if re.match(r"^[A-E]$", input_string, re.IGNORECASE):
        return input_string.upper(), ""

    match = re.search(r"Answer:\s*([A-E])\)?\.?\s*(.*)", input_string, re.IGNORECASE)
    if match:
        option_number = match.group(1).upper()  # Normalize option number to uppercase
        option_text = match.group(2).strip()
        return option_number, option_text
    else:
        # If no option number is found after 'Answer:', consider it as no valid answer provided
        return None, input_string.strip()


def evaluate_semantic_similarity(response, answer_key_number, answer_key_text, normalize_fn):
    """
    Evaluates whether the answer key and student response are semantically the same.
    Returns a score of 1 if they match, otherwise 0.
    """
    student_response_number, student_response_text = eval(normalize_fn)(response)

    # Compare option numbers and option texts (if available) to determine a match
    if answer_key_number and student_response_number:
        if answer_key_number == student_response_number:
            if answer_key_text and student_response_text:
                # If both strings have option texts, they must match as well
                return (1, student_response_number, student_response_text) if answer_key_text.lower() == student_response_text.lower() else (0, student_response_number, student_response_text)
            # If only option numbers are provided or one string lacks option text, it's a match
            return (1, student_response_number, student_response_text)
    elif answer_key_text.lower() == student_response_text.lower():
        # If no option numbers are present, but the option texts match, it's also considered a match
        return (1, student_response_number, student_response_text)

    return (0, student_response_number, student_response_text)


def eval_response(response, answer_key_number, answer_key_text, normalize_fn="normalize_string"):
    return evaluate_semantic_similarity(response, answer_key_number, answer_key_text, normalize_fn)


def cinepile_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case accuracy), value: metric value
    """
    pred = results[0]

    ans_key_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
    answer = ans_key_map[doc["answer_key_position"]]
    correct, response_number, response_text = eval_response(pred, answer, doc["answer_key"])

    question_category = doc["question_category"]
    hard_split = doc["hard_split"]

    data_dict = {
        "question": doc["question"],
        "question_category": question_category,
        "hard_split": hard_split,
        "correct": correct,
        "answer": answer,
        "raw_response": pred,
        "response_ext_number": response_number,
        "response_ext_text": response_text,
    }

    return {"cinepile_accuracy": data_dict}


CATEGORIES = ["Character and\nRelationship Dynamics", "Narrative and\nPlot Analysis", "Setting and\nTechnical Analysis", "Temporal", "Theme Exploration"]
HARD_SPLIT = ["True", "False"]


def cinepile_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """

    cat2score = defaultdict(dict)
    for result in results:
        score = result["correct"]
        ques_category = result["question_category"]
        hard_split = result["hard_split"]
        if ques_category not in cat2score:
            cat2score[ques_category] = defaultdict(dict)

            cat2score[ques_category][HARD_SPLIT[0]] = {"correct": 0, "answered": 0}
            cat2score[ques_category][HARD_SPLIT[1]] = {"correct": 0, "answered": 0}

        cat2score[ques_category][hard_split]["answered"] += 1
        cat2score[ques_category][hard_split]["correct"] += score

    total_correct, total_answered = 0, 0
    for category in CATEGORIES:
        for hard_split in HARD_SPLIT:
            total_correct += cat2score[category][hard_split]["correct"]
            total_answered += cat2score[category][hard_split]["answered"]

    aggregate_accuracy = 100 * total_correct / total_answered if total_answered > 0 else 0
    eval_logger.info(f"Overall Performance: {aggregate_accuracy: .1f}%")

    for category in CATEGORIES:
        category_correct, category_answered = 0, 0
        for hard_split in HARD_SPLIT:
            category_correct += cat2score[category][hard_split]["correct"]
            category_answered += cat2score[category][hard_split]["answered"]
        category_accuracy = 100 * category_correct / category_answered if category_answered > 0 else 0
        eval_logger.info(f"\t{category} Acc: {category_accuracy:.1f}%")
        for hard_split in HARD_SPLIT:
            correct = cat2score[category][hard_split]["correct"]
            answered = cat2score[category][hard_split]["answered"]
            accuracy = 100 * correct / answered if answered > 0 else 0
            eval_logger.info(f"\t\t{category} Hard {hard_split}: {accuracy:.1f}%")

    return aggregate_accuracy
