import logging
import yaml
import os
from pathlib import Path
import pandas as pd
import json

eval_logger = logging.getLogger("lmms-eval")
from lmms_eval.tasks.mathverse.mathverse_evals import MathVerseEvaluator
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

with open(Path(__file__).parent / "mathverse.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

mathverse_evaluator = MathVerseEvaluator(api_key=os.getenv("OPENAI_API_KEY", "YOUR_API_KEY"), gpt_model=config["metadata"]["gpt_eval_model_name"])


def mathverse_doc_to_visual(doc):
    if str(doc["image"]).strip() == "":
        return []
    return [doc["image"].convert("RGB")]


def mathverse_doc_to_text(doc, model_specific_prompt_kwargs=None):
    problem = {
        "question": doc["question"],
        "answer": doc["answer"] if "answer" in doc else None,
        "query_wo": doc["query_wo"],
        "query_cot": doc["query_cot"],
        "question_type": doc["question_type"],
        "problem_version": doc["problem_version"],
    }
    query_prompt = mathverse_evaluator.create_one_query(
        problem, examples=None, shot_num=0, shot_type=model_specific_prompt_kwargs["shot_type"], hint=model_specific_prompt_kwargs.get("hint", None), query_type=model_specific_prompt_kwargs["query_type"]
    )
    return query_prompt


def mathverse_process_results(doc, results):
    prediction = results[0].strip()

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
    }

    return {
        "gpt_eval_score": result,
        "submission": result,
    }


def mathverse_aggregate_results_submission(results, args, *, calculate_gain=False, random_scores=None):
    split_flag = results[0]["metadata"]["split"]
    path = generate_submission_file(f"mathverse_{split_flag}_results.json", args)
    with open(path, "w") as f:
        json.dump(results, f, indent=4)

    eval_logger.info(f"Saved results to {path}")


def mathverse_aggregate_results_eval(results, args, *, calculate_gain=False, random_scores=None):
    split_flag = results[0]["metadata"]["split"]
    # save the result first, in case the gpt evaluation fails
    path = generate_submission_file(f"mathverse_{split_flag}_results.json", args)
    with open(path, "w") as f:
        json.dump(results, f, indent=4)
    # gpt evaluation
    results_dict, scores = mathverse_evaluator.eval_results(results, config)
    # save results
    path = generate_submission_file(f"mathverse_{split_flag}_results.json", args)
    with open(path, "w") as f:
        json.dump(results_dict, f, indent=4)
    # save scores
    path = generate_submission_file(f"mathverse_{split_flag}_scores.json", args)
    with open(path, "w") as f:
        json.dump(scores, f, indent=4)
    eval_logger.info(f"Saved scores to {path}")
    if scores["average"]["accuracy"] == 0:
        return None
    return scores["average"]["accuracy"]
