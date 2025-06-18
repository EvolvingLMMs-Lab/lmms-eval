import json
import os
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger as eval_logger

from lmms_eval.llm_judge import ServerConfig, get_server
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.mathverse.mathverse_evals import MathVerseEvaluator

with open(Path(__file__).parent / "mathverse.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

# Initialize the judge server
API_TYPE = os.getenv("API_TYPE", "openai")
GPT_MODEL = os.getenv("MODEL_VERSION", config["metadata"]["gpt_eval_model_name"])

server_config = ServerConfig(
    model_name=GPT_MODEL,
)
server = get_server(server_name=API_TYPE, config=server_config)

mathverse_evaluator = MathVerseEvaluator(api_key=os.getenv("OPENAI_API_KEY", "YOUR_API_KEY"), gpt_model=config["metadata"]["gpt_eval_model_name"])


def mathverse_doc_to_visual(doc):
    if str(doc["image"]).strip() == "":
        return []
    return [doc["image"].convert("RGB")]


def mathverse_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    problem = {
        "question": doc["question"],
        "answer": doc["answer"] if "answer" in doc else None,
        "query_wo": doc["query_wo"],
        "query_cot": doc["query_cot"],
        "question_type": doc["question_type"],
        "problem_version": doc["problem_version"],
    }
    query_prompt = mathverse_evaluator.create_one_query(
        problem, examples=None, shot_num=0, shot_type=lmms_eval_specific_kwargs["shot_type"], hint=lmms_eval_specific_kwargs.get("hint", None), query_type=lmms_eval_specific_kwargs["query_type"]
    )
    return query_prompt


def mathverse_process_results(doc, results):
    prediction = results[0].strip()
    question = doc["question_for_eval"]
    answer = doc["answer"] if "answer" in doc else None

    # Define custom prompt for MathVerse evaluation
    custom_prompt = """Below are two answers to a math question. Determine whether these two answers are consistent.
Please note that only when the Model Answer completely matches the Standard Answer means they are consistent. For non-multiple-choice questions, if the meaning is expressed in the same way, it is also considered consistent, for example, 0.5m and 50cm.

Return only "Yes" if they are consistent or "No" if they are different.
Only return "Yes" or "No" with no additional text or formatting."""

    judge_result = 0
    if answer is not None:
        try:
            # Use the llm_judge API for binary evaluation
            result = server.evaluate_binary(question=question, answer=str(answer), prediction=prediction, output_format="yes/no", custom_prompt=custom_prompt)

            # Parse the result
            if result["success"]:
                judge_response = result["result"]
                judge_result = 1 if judge_response and judge_response.lower() == "yes" else 0
            else:
                eval_logger.error(f"Judge evaluation failed: {result.get('raw_response', 'Unknown error')}")
                judge_result = 0

        except Exception as e:
            eval_logger.error(f"Error getting judge response: {e}")
            judge_result = 0

    result = {
        "sample_index": doc["sample_index"],
        "problem_index": doc["problem_index"],
        "problem_version": doc["problem_version"],
        "question": doc["question"],
        "answer": doc["answer"] if "answer" in doc else None,
        "prediction": prediction,
        "question_type": doc["question_type"],
        "metadata": doc["metadata"],
        "query_wo": doc["query_wo"],
        "query_cot": doc["query_cot"],
        "question_for_eval": doc["question_for_eval"],
        "true_false": judge_result == 1,
    }

    return {
        "llm_as_judge_eval": judge_result,
        "gpt_eval_score": result,
        "submission": result,
    }


def mathverse_aggregate_results_submission(results, args, *, calculate_gain=False, random_scores=None):
    # Don't know why but this sometimes yields error so I hardcode it
    try:
        split_flag = results[0]["metadata"]["split"]
    except:
        split_flag = "testmini"
    path = generate_submission_file(f"mathverse_{split_flag}_results.json", args)
    with open(path, "w") as f:
        json.dump(results, f, indent=4)

    eval_logger.info(f"Saved results to {path}")


def mathverse_aggregate_results_eval(results, args, *, calculate_gain=False, random_scores=None):
    split_flag = results[0]["metadata"]["split"]
    problem_version = results[0]["problem_version"].lower().replace(" ", "_")
    # save the result first, in case the gpt evaluation fails
    path = generate_submission_file(f"mathverse_{split_flag}_{problem_version}_results.json", args)
    with open(path, "w") as f:
        json.dump(results, f, indent=4)
    # gpt evaluation
    results_dict, scores = mathverse_evaluator.eval_results(results, config)
    # save results
    path = generate_submission_file(f"mathverse_{split_flag}_{problem_version}_results.json", args)
    with open(path, "w") as f:
        json.dump(results_dict, f, indent=4)
    # save scores
    path = generate_submission_file(f"mathverse_{split_flag}_{problem_version}_scores.json", args)
    with open(path, "w") as f:
        json.dump(scores, f, indent=4)
    eval_logger.info(f"Saved scores to {path}")
    if scores["average"]["accuracy"] == 0:
        return None
    return scores["average"]["accuracy"]
