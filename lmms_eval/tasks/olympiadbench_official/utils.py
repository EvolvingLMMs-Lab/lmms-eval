# Copyright 2025 Xiaomi Corporation.

import datetime
import json
import os
import re

from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks._task_utils.math_verify_utils import MathVerifyFn
from lmms_eval.tasks.olympiadbench_official.olympiadbench_evals import (
    OlympiadBenchEvaluator,
)

dir_name = os.path.dirname(os.path.abspath(__file__))

olympiadbench_evaluator = OlympiadBenchEvaluator()


def olympiadbench_doc_to_visual(doc):
    res = []
    for i in range(1, 6):
        image_key = f"image_{i}"
        if doc[image_key] is not None:
            res.append(doc[image_key])
    return [image.convert("RGB") for image in res]


def olympiadbench_doc_to_text(doc):
    question = doc["question"]
    subject = doc["subfield"]
    mul_ans = doc["is_multiple_answer"]
    if mul_ans is None:
        mul_ans = False
    ans_type = doc["answer_type"]
    if ans_type == "Need_human_evaluate":
        ans_type = "proof based"

    language = doc["language"]
    if language == "English":
        pre_prompt = f"The following is a question from an International {subject} competition.\n"

        post_prompt = ""
        if not mul_ans:
            post_prompt += f"The answer of the question should be {ans_type}.\n"
        else:
            post_prompt += f"The question has multiple answers, each of them should be {ans_type}.\n"
        post_prompt += "Please calculate the answer according to the given requirements and the information provided. Please use LaTeX format to represent the variables and formulas used in the solution process and results. Please put your solution in "
        if not mul_ans:
            post_prompt += "\\boxed{}"
        else:
            post_prompt += "\\boxed{MULTIPLE ANSWERS CONNECTED WITH COMMAS}"

    elif language == "Chinese":
        pre_prompt = f"以下是中国{subject}竞赛中的解答题。\n"

        post_prompt = ""
        if not mul_ans:
            post_prompt += f"答案类型为{ans_type}。\n"
        else:
            post_prompt += f"题目有多个答案，答案类型均为{ans_type}。\n"
        post_prompt += "请根据题目的要求和所提供的信息计算得出答案。解答过程和结果中使用的变量和公式请使用LaTeX格式表示。请把答案放在"
        if not mul_ans:
            post_prompt += "\\boxed{}"
        else:
            post_prompt += "\\boxed{用英⽂逗号连接的多个答案}"

    final_question = pre_prompt + question + "\n" + post_prompt
    return final_question


math_verify_fn = MathVerifyFn()


def olympiadbench_process_results(doc, results):
    precision = doc["error"]
    is_proving = doc["question_type"] == "Theorem proof" or doc["final_answer"] is None
    assert not is_proving, "Proof based questions are not supported"
    if precision is None:
        precision = 0
    prediction = results[0].strip()

    gt = doc["final_answer"][0].rstrip("。")

    ext_prediction = re.findall(r"(\\boxed\{.*\})", prediction)
    if len(ext_prediction) == 0:
        ext_prediction = ""
    else:
        ext_prediction = ext_prediction[0]
    accuracy = olympiadbench_evaluator.judge(ext_prediction, gt, precision)
    accuracy = int(accuracy)

    math_verify_score, math_verify_ext = math_verify_fn(prediction, gt)

    category = f"{doc['subject']}_{doc['language']}"
    return {
        "exact_match": accuracy,
        "math_verify": {"score": math_verify_score, "extraction": math_verify_ext},
        category: {"accuracy": accuracy, "language": doc["language"], "subject": doc["subject"]},
    }


def olympiadbench_aggregate_results(results, args):
    results = [item["accuracy"] for item in results]
    return sum(results) / len(results)


def olympiadbench_math_verify_aggregate_results(results, args):
    total = len(results)
    score = sum(result["score"] for result in results)
    return score / total
