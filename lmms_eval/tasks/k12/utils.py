import datetime
import json
import os

from loguru import logger as eval_logger

from lmms_eval.api.judge_config_helper import create_judge
from lmms_eval.api.judge_utils import JudgePromptBuilder

dir_name = os.path.dirname(os.path.abspath(__file__))

# Initialize judge for K12 evaluations using environment-based configuration
k12_judge = create_judge(default_model="gpt-4o-2024-11-20", temperature=0.5)


def build_zh_exam_k12_gpt4_prompt(question_data):
    """Build prompt using unified prompt builder"""
    question = question_data["question"]
    answer = question_data["answer"]
    response = str(question_data["response"])

    # Use unified prompt builder for correctness evaluation
    return JudgePromptBuilder.build_correctness_prompt(question=question, answer=answer, prediction=response, output_format="yes/no")


def k12_doc_to_visual(doc):
    visual_list = []
    if "image" in doc and doc["image"] is not None:
        visual_list.append(doc["image"].convert("RGB"))
    return visual_list


def k12_doc_to_text(doc):
    question = doc["question"]
    return question


def k12_process_results(doc, results):
    prediction = results[0].strip()
    # Build the prompt for GPT-4o evaluation
    question_data = {"id": doc.get("id", "unknown"), "question": doc["question"], "answer": doc["answer"], "response": prediction}

    # Use unified judge API for evaluation
    try:
        result = k12_judge.evaluate_binary(question=doc["question"], answer=doc["answer"], prediction=prediction, output_format="yes/no")

        judge_result = 1 if result["result"] else 0
    except Exception as e:
        eval_logger.error(f"Error getting judge response: {e}")
        judge_result = 0

    return {"llm_as_judge_eval": judge_result}
