import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml
from loguru import logger as eval_logger

from lmms_eval.llm_judge import ServerConfig, get_server
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

hf_home = os.getenv("HF_HOME", "~/.cache/huggingface")

base_cache_dir = os.path.expanduser(hf_home)


with open(Path(__file__).parent / "mmvu_val.yaml", "r") as f:
    raw_data_val = f.readlines()
    safe_data_val = []
    for i, line in enumerate(raw_data_val):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data_val.append(line)
cache_name_val = yaml.safe_load("".join(safe_data_val))["dataset_kwargs"]["cache_dir"]
cache_dir_val = os.path.join(base_cache_dir, cache_name_val)

# Initialize the LLM judge server
API_TYPE = os.getenv("API_TYPE", "openai")
MODEL_VERSION = os.getenv("MODEL_VERSION", "gpt-4o-2024-08-06")

server_config = ServerConfig(
    model_name=MODEL_VERSION,
)
server = get_server(server_name=API_TYPE, config=server_config)


def mmvu_doc_to_visual_val(doc):
    video_path = doc["video_path"]
    video_path = os.path.join(cache_dir_val, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


multiple_choice_prompt = """
Question:{question}
A: {a}
B: {b}
C: {c}
D: {d}
E: {e}
Visual Information: processed video
Do not generate any intermediate reasoning process. Answer directly with the option letter from the
given choices.
"""

open_ended_prompt = """
Question:{question}
Visual Information: processed video
Do not generate any intermediate reasoning process. Directly output the final answer.
"""

multiple_choice_prompt_cot = """
Question:{question}
A: {a}
B: {b}
C: {c}
D: {d}
E: {e}
Visual Information: processed video
Answer the given multiple-choice question step by step. Begin by explaining your reasoning process
clearly. Conclude by stating the final answer using the following format: "Therefore, the final answer
is: $LETTER" (without quotes), where $LETTER is one of the options. Think step by step before
answering.
"""

open_ended_prompt_cot = """
Question:{question}
Visual Information: processed video
Answer the given question step by step. Begin by explaining your reasoning process clearly. Conclude
by stating the final answer using the following format: "Therefore, the final answer is: "Answer:
$ANSWER" (without quotes), where $ANSWER is the final answer of the question. Think step by
step before answering.
"""


def mmvu_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question_type = doc["question_type"]
    if question_type == "multiple-choice":
        question = doc["question"]
        choices = doc["choices"]
        full_prompt = multiple_choice_prompt.format(question=question, a=choices["A"], b=choices["B"], c=choices["C"], d=choices["D"], e=choices["E"])
    else:
        question = doc["question"]
        full_prompt = open_ended_prompt.format(question=question)
    return full_prompt


def mmvu_doc_to_text_cot(doc, lmms_eval_specific_kwargs=None):
    question_type = doc["question_type"]
    if question_type == "multiple-choice":
        question = doc["question"]
        choices = doc["choices"]
        full_prompt = multiple_choice_prompt_cot.format(question=question, a=choices["A"], b=choices["B"], c=choices["C"], d=choices["D"], e=choices["E"])
    else:
        question = doc["question"]
        full_prompt = open_ended_prompt_cot.format(question=question)
    return full_prompt


def construct_question_prompt(doc):
    """Construct the question prompt for evaluation"""
    question_type = doc["question_type"]
    if question_type == "multiple-choice":
        choices = doc["choices"]
        return f"""Question: {doc["question"]}
A: {choices["A"]}
B: {choices["B"]}
C: {choices["C"]}
D: {choices["D"]}
E: {choices["E"]}"""
    else:
        return f"Question: {doc['question']}"


def evaluate_with_llm_judge(doc, prediction):
    """Use the LLM judge interface to evaluate the prediction"""
    formatted_question = construct_question_prompt(doc)
    answer = doc["answer"]
    question_type = doc["question_type"]

    if question_type == "multiple-choice":
        # For multiple choice, construct full answer including the choice content
        if answer in doc["choices"]:
            full_answer = f"{answer}: {doc['choices'][answer]}"
        else:
            full_answer = str(answer)

        custom_prompt = """You are a strict evaluator assessing answer correctness. You must output 1 for fully correct answers and 0 for any other case.

# Evaluation Rules for Multiple Choice Questions
- The model prediction may contain reasoning, but focus on the final answer.
- Score 1 if the predicted answer matches the ground truth answer.
- The answer can be given as just the letter (A, B, C, D, E) or include the full option text.
- Ignore minor differences in formatting, capitalization, or spacing.
- Score 0 for any incorrect answer, even if the reasoning process seems correct.

Return only "1" or "0" with no additional text or formatting."""
    else:
        # For open-ended questions
        full_answer = str(answer)

        custom_prompt = """You are a strict evaluator assessing answer correctness. You must output 1 for fully correct answers and 0 for any other case.

# Evaluation Rules for Open-Ended Questions
- The model prediction may contain reasoning, focus on extracting the final answer.
- Score 1 if the prediction matches the answer semantically, even if in different format.
- Score 0 for partially correct answers or answers with extra incorrect information.
- Ignore minor differences in formatting, capitalization, or spacing.
- Treat numerical answers as correct if they match within reasonable precision.
- For questions requiring units, both value and unit must be correct.

Return only "1" or "0" with no additional text or formatting."""

    try:
        # Use the llm_judge API for binary evaluation
        result = server.evaluate_binary(question=formatted_question, answer=full_answer, prediction=prediction, output_format="0/1", custom_prompt=custom_prompt)

        # Parse the result
        if result["success"]:
            judge_response = result["result"]
            judge_score = judge_response.strip()
        else:
            eval_logger.error(f"Judge evaluation failed: {result.get('raw_response', 'Unknown error')}")
            judge_score = "0"

    except Exception as e:
        eval_logger.error(f"Error getting judge response: {e}")
        judge_score = "0"

    return judge_score == "1"


def extract_category(doc):
    category = doc["video_path"].split("/")[-2]
    return category


def mmvu_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case videomme score), value: metric value
    """
    # Handle the case where results[0] might be a list or a string
    pred = results[0]
    if isinstance(pred, list):
        pred_ans = pred[0] if pred else ""
    else:
        pred_ans = pred

    # Ensure pred_ans is a string
    pred_ans = str(pred_ans)

    category = extract_category(doc)

    # Use the new LLM judge interface for evaluation
    correct = evaluate_with_llm_judge(doc, pred_ans)

    # Extract predicted answer for logging (best effort)
    if doc["question_type"] == "multiple-choice":
        # Try to extract the letter choice from the prediction
        import re

        letter_match = re.search(r"\b([A-E])\b", pred_ans)
        extracted_answer = letter_match.group(1) if letter_match else "N/A"
    else:
        # For open-ended, just use the prediction as-is (truncated for logging)
        extracted_answer = pred_ans[:100] + "..." if len(pred_ans) > 100 else pred_ans

    data_dict = {"question_id": doc["id"], "category": category, "pred_answer": extracted_answer, "answer": doc["answer"], "correct": int(correct)}

    return {f"accuracy": data_dict}


def mmvu_aggregate_results_val(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """

    TASK_MAP = {
        "Biology": "Science",
        "Chemistry": "Science",
        "Modern_Physics": "Science",
        "Astronomy": "Science",
        "Geography": "Science",
        "Materials_Science": "Science",
        "Neurobiology": "Science",
        "Electromagnetism": "Science",
        "Thermodynamics": "Science",
        "Mechanics": "Science",
        "Civil_Engineering": "Engineering",
        "Electrical_Engineering": "Engineering",
        "Mechanical_Engineering": "Engineering",
        "Biomedical_Engineering": "Engineering",
        "Electronics_and_Communication": "Engineering",
        "Computer_Science": "Engineering",
        "Clinical_Medicine": "Healthcare",
        "Basic_Medicine": "Healthcare",
        "Preventive_Medicine": "Healthcare",
        "Pharmacy": "Healthcare",
        "Dentistry": "Healthcare",
        "Art": "Humanities_and_Social_Science",
        "Literature": "Humanities_and_Social_Science",
        "History": "Humanities_and_Social_Science",
        "Law": "Humanities_and_Social_Science",
        "Economics": "Humanities_and_Social_Science",
        "Management": "Humanities_and_Social_Science",
    }

    TASK_TYPES = list(set(TASK_MAP.values()))

    category2score = {}
    for task_type in TASK_TYPES:
        category2score[task_type] = {"correct": 0, "answered": 0}

    for result in results:
        category = result["category"]
        if category in TASK_MAP:
            category = TASK_MAP[category]
            category2score[category]["answered"] += 1
            category2score[category]["correct"] += result.get("correct", False)
    category_scores = {}

    for category in TASK_TYPES:
        total_correct = category2score[category]["correct"]
        total_answered = category2score[category]["answered"]
        accuracy = 100 * total_correct / total_answered if total_answered > 0 else 0
        category_scores[category] = accuracy

    total_correct = sum(category2score[category]["correct"] for category in TASK_TYPES)
    total_answered = sum(category2score[category]["answered"] for category in TASK_TYPES)
    accuracy = 100 * total_correct / total_answered if total_answered > 0 else 0
    eval_logger.info("=" * 50)
    eval_logger.info(f"Average Accuracy: {accuracy:.2f}%")
    eval_logger.info("Categorical accuracy: ")
    for key, value in category_scores.items():
        eval_logger.info(f"{key} accuracy: {value:.2f}%")
    eval_logger.info("=" * 50)
    return accuracy
