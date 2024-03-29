import os
import json
import datetime
from lmms_eval.tasks.olympiadbench.olympiadbench_evals import OlympiadBenchEvaluator
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

import logging
eval_logger = logging.getLogger("lmms-eval")
dir_name = os.path.dirname(os.path.abspath(__file__))

olympiadbench_evaluator = OlympiadBenchEvaluator()

def olympiadbench_doc_to_visual(doc):
    return [image.convert("RGB") for image in doc["images"]]

def olympiadbench_doc_to_text(doc):
    question = doc["question"]
    subject = doc["subfield"]
    mul_ans = doc["is_multiple_answer"]
    if mul_ans is None:
        mul_ans = False
    ans_type = doc["answer_type"]
    if ans_type == "Need_human_evaluate":
        ans_type = "proof based"

    pre_prompt = f"The following is a question from an International {subject} competition.\n"

    post_prompt = ""
    if not mul_ans:
        post_prompt += f"The answer of the question should be {ans_type}.\n"
    else:
        post_prompt += f"The question has multiple answers, each of them should be {ans_type}.\n"
    post_prompt += "Please calculate the answer according to the given requirements and the information provided. Please use LaTeX format to represent the variables and formulas used in the solution process and results. Please end your solution with "
    if not mul_ans:
        post_prompt += '"So the final answer is \\boxed{answer}."\n'
    else:
        post_prompt += 'So the final answer is \\boxed{multiple answers connected with commas}.\n'

    final_question = pre_prompt + question + '\n' + post_prompt
    return final_question

def olympiadbench_process_results(doc, results):
    precision = doc["error"]
    is_proving = "TP" in doc["source"] 
    if precision is None:
        precision = 0
    prediction = results[0].strip()

    if is_proving:
        return {
            "submission": prediction
        }
    else:
        prediction = prediction.split("final answer is")[-1]
        prediction = prediction.replace('"', "").replace("\n", "").replace(" ", "").strip(".").strip("。")
        accuracy = olympiadbench_evaluator.judge(prediction, doc["final_answer"][0], precision)
        accuracy = int(accuracy)
        return {
            "exact_match": accuracy
        }

def olympiadbench_aggregate_results(results, args):
    now_date_time = datetime.datetime.now().strftime("%Y-%m%d-%H%M-%S")
    submission_file_name = f"olympiadbench-test-en-submission-{now_date_time}.json"
    path = generate_submission_file(submission_file_name, args)
    with open(path, "w") as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"Submission file saved to {path}")
    