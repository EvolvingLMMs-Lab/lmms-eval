import io
import json
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger as eval_logger
from PIL import Image

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.mmrefine.mmrefine_evals import MMRefineEvaluator

with open(Path(__file__).parent / "mmrefine.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))


mmrefine_evaluator = MMRefineEvaluator()
if config.get("metadata", {}).get("gpt_eval_model_name"):
    mmrefine_evaluator.gpt_model = config["metadata"]["gpt_eval_model_name"]


def mmrefine_doc_to_visual(doc):
    if doc["image"] is None or str(doc["image"]).strip() == "":
        return []
    if isinstance(doc["image"], Image.Image):
        return [doc["image"].convert("RGB")]
    return [Image.open(io.BytesIO(doc["image"]["bytes"])).convert("RGB")]


def mmrefine_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    problem = {
        "id": doc["id"],
        "question": doc["question"],
        "answer": doc["answer"] if "answer" in doc else None,
        "initial_solution": doc["initial_solution"],
        "solution_source": doc["solution_source"],
        "solution_label": doc["solution_label"],
        "reference_feedback": doc["reference_feedback"] if "reference_feedback" in doc else None,
        "meta": doc["meta"],
    }
    query_prompt = mmrefine_evaluator.create_one_query(problem)
    return query_prompt


def mmrefine_process_results(doc, results):
    prediction = results[0].strip()

    result = {
        "id": doc["id"],
        "question": doc["question"],
        "answer": doc["answer"] if "answer" in doc else None,
        "prediction": prediction,
        "initial_solution": doc["initial_solution"],
        "solution_source": doc["solution_source"],
        "solution_label": doc["solution_label"],
        "reference_feedback": doc["reference_feedback"] if "reference_feedback" in doc else None,
        "meta": doc["meta"],
    }
    result = mmrefine_evaluator.eval_result(result, config)

    metric_dict = {}
    for _metric in [
        "Refinement Failure",
        "Error Detection Success",
        "Error Correction Success",
        "Refinement Success",
        "False Error Detection",
        "Validation Success",
        "RefScore",
        "mRecall",
    ]:
        _result = result.copy()
        metric_dict[_metric] = _result
    return metric_dict


def mmrefine_aggregate_results(results, args=None, **kwargs):
    # calculate total scores
    results_dict = {f"{result['id']}_{result['solution_source']}": result for result in results}
    results = pd.DataFrame(results)
    try:
        assert len(results) == len(results_dict)
    except AssertionError:
        eval_logger.error(f"Length mismatch: {len(results)} results vs {len(results_dict)} unique entries")

    n_correct_solutions = len(results[results["solution_label"] == "correct"])
    n_incorrect_solutions = len(results) - n_correct_solutions

    vc = results["eval_result"].value_counts()
    scores = {
        "Refinement Failure": vc.loc["Refinement Failure"] / n_incorrect_solutions if "Refinement Failure" in vc.keys() else 0.0,
        "Error Detection Success": vc.loc["Error Detection Success"] / n_incorrect_solutions if "Error Detection Success" in vc.keys() else 0.0,
        "Error Correction Success": vc.loc["Error Correction Success"] / n_incorrect_solutions if "Error Correction Success" in vc.keys() else 0.0,
        "Refinement Success": vc.loc["Refinement Success"] / n_incorrect_solutions if "Refinement Success" in vc.keys() else 0.0,
        "False Error Detection": vc.loc["False Error Detection"] / n_correct_solutions if "False Error Detection" in vc.keys() else 0.0,
        "Validation Success": vc.loc["Validation Success"] / n_correct_solutions if "Validation Success" in vc.keys() else 0.0,
    }

    scores["RefScore"] = scores["Refinement Success"] - scores["False Error Detection"]
    scores["Error Correction Success"] = scores["Error Correction Success"] + scores["Refinement Success"]
    scores["Error Detection Success"] = scores["Error Detection Success"] + scores["Error Correction Success"]
    scores["mRecall"] = (scores["Error Detection Success"] + (1 - scores["False Error Detection"])) / 2

    scores["results_dict"] = results_dict

    return scores


def mmrefine_aggregate_results_refinement_failure(results, args=None, **kwargs):
    return mmrefine_aggregate_results(results, args, **kwargs)["Refinement Failure"]


def mmrefine_aggregate_results_error_detection_success(results, args=None, **kwargs):
    return mmrefine_aggregate_results(results, args, **kwargs)["Error Detection Success"]


def mmrefine_aggregate_results_error_correction_success(results, args=None, **kwargs):
    return mmrefine_aggregate_results(results, args, **kwargs)["Error Correction Success"]


def mmrefine_aggregate_results_refinement_success(results, args=None, **kwargs):
    return mmrefine_aggregate_results(results, args, **kwargs)["Refinement Success"]


def mmrefine_aggregate_results_false_error_detection(results, args=None, **kwargs):
    return mmrefine_aggregate_results(results, args, **kwargs)["False Error Detection"]


def mmrefine_aggregate_results_validation_success(results, args=None, **kwargs):
    return mmrefine_aggregate_results(results, args, **kwargs)["Validation Success"]


def mmrefine_aggregate_results_refscore(results, args=None, **kwargs):
    scores = mmrefine_aggregate_results(results, args, **kwargs)
    results_dict = scores.pop("results_dict")

    path = generate_submission_file("mmrefine_results.json", args)
    with open(path, "w") as f:
        json.dump(results_dict, f, indent=4)

    path = generate_submission_file("mmrefine_scores.json", args)
    with open(path, "w") as f:
        json.dump(scores, f, indent=4)
    return scores["RefScore"]


def mmrefine_aggregate_results_mrecall(results, args=None, **kwargs):
    return mmrefine_aggregate_results(results, args, **kwargs)["mRecall"]
