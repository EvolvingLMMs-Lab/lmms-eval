import json
import os
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.reasoning_utils import compute_score
from lmms_eval.tasks.mathvista.mathvista_evals import MathVistaEvaluator

SYSTEM_PROMPT = (
    "You are a helpful assistant. When the user asks a question, your response must include two parts: "
    "first, the reasoning process enclosed in <think>...</think> tags, then the final answer enclosed in <answer>...</answer> tags."
    "Please provide a clear, concise response within <answer> </answer> tags that directly addresses the question."
)

with open(Path(__file__).parent / "mathvista_testmini_cot.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))


mathvista_evaluator = MathVistaEvaluator()


def mathvista_doc_to_visual(doc):
    return [doc["decoded_image"].convert("RGB")]


def mathvista_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    problem = {
        "question_type": doc["question_type"],
        "answer_type": doc["answer_type"],
        "question": doc["question"],
        "unit": doc["unit"] if "unit" in doc else "",
        "caption": doc["caption"] if "caption" in doc else "",
        "ocr": doc["ocr"] if "ocr" in doc else "",
        "choices": doc["choices"],
        "answer": doc["answer"] if "answer" in doc else None,
        "precision": doc["precision"] if "precision" in doc else 0,
    }
    query_prompt = mathvista_evaluator.create_one_query(
        problem,
        shot_num=lmms_eval_specific_kwargs["shot"],
        shot_type=lmms_eval_specific_kwargs["shot_type"],
        use_caption=lmms_eval_specific_kwargs["use_caption"],
        use_ocr=lmms_eval_specific_kwargs["use_ocr"],
    )
    return query_prompt


def mathvista_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    query_prompt = mathvista_doc_to_text(doc, lmms_eval_specific_kwargs)
    system_messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
    visuals = mathvista_doc_to_visual(doc)

    user_messages = []
    user_messages.append({"role": "user", "content": [{"type": "image", "url": visuals[0]}, {"type": "text", "text": query_prompt}]})

    return system_messages + user_messages


def mathvista_process_results(doc, results):
    acc_score = 0
    format_score = 0
    # Build question text for extra_info using task's doc_to_text
    try:
        default_kwargs = config.get("lmms_eval_specific_kwargs", {}).get("default", {})
    except Exception:
        default_kwargs = {}
    question = mathvista_doc_to_text(doc, default_kwargs)
    extra_info = {"question": question}
    for pred in results:
        options = ["A", "B", "C", "D", "E", "F", "G", "H"]
        choices = doc["choices"]
        if choices and doc["question_type"] == "multi_choice":
            choice_index = choices.index(doc["answer"])
            ground_truth = options[choice_index]
        else:
            ground_truth = doc["answer"]
        score_dict = compute_score(data_source="mathvista", solution_str=pred.strip(), ground_truth=ground_truth, extra_info=extra_info)
        acc_score += score_dict["acc_score"]
        format_score += score_dict.get("format_reward_score", 0.0)

    return {"acc_score": acc_score / len(results) if results else 0.0, "format_score": format_score / len(results) if results else 0.0}


def mathvista_aggregate_results(results, args, *, calculate_gain=False, random_scores=None):
    split_flag = results[0]["metadata"]["split"]
    full_pids = [result["question_id"] for result in results]
    total = len(results)
    correct = sum(1 for idx, pid in enumerate(full_pids) if results[idx]["true_false"])
    accuracy = round(correct / total * 100, 2)
    scores = {"average": {"accuracy": accuracy, "correct": correct, "total": total}}

    for result in results:
        result.update(result.pop("metadata"))

    results_dict = {result["question_id"]: result for result in results}
    df = pd.DataFrame(results_dict).T
    target_keys = ["question_type", "answer_type", "language", "source", "category", "task", "context", "grade", "skills"]

    for key in target_keys:
        values = df[key].explode().unique() if key == "skills" else df[key].unique()
        scores[key] = {}
        for value in values:
            correct, total, acc = mathvista_evaluator.get_acc_with_contion(df, key, value)
            if total > 0:
                scores[key][value] = {"accuracy": acc, "correct": correct, "total": total}
        scores[key] = dict(sorted(scores[key].items(), key=lambda item: float(item[1]["accuracy"]), reverse=True))

    if calculate_gain:
        for key in scores:
            if key == "average":
                gain = round(float(scores[key]["accuracy"]) - float(random_scores[key]["accuracy"]), 2)
                scores[key]["acc_gain"] = gain
            else:
                for sub_key in scores[key]:
                    gain = round(float(scores[key][sub_key]["accuracy"]) - float(random_scores[key][sub_key]["accuracy"]), 2)
                    scores[key][sub_key]["acc_gain"] = gain

    if scores["average"]["accuracy"] == 0:
        return None
    return scores["average"]["accuracy"]
