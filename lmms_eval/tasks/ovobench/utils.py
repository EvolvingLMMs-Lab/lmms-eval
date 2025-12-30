import json
import os

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.ovobench.constant import (
    BR_PROMPT_TEMPLATE,
    CRR_PROMPT_TEMPLATE,
    REC_PROMPT_TEMPLATE,
    SSR_PROMPT_TEMPLATE,
)
from lmms_eval.tasks.ovobench.score_utils.score import (
    calculate_score_backward_realtime,
    calculate_score_forward,
)


def is_forward_task(doc):
    task = doc["task"]
    return not task in ["EPM", "ASI", "HLD", "STU", "OJR", "ATR", "ACR", "OCR", "FPD"]


def get_task_type(task_name):
    backward_tasks = ["EPM", "ASI", "HLD"]
    realtime_tasks = ["STU", "OJR", "ATR", "ACR", "OCR", "FPD"]
    forward_tasks = ["REC", "SSR", "CRR"]

    if task_name in backward_tasks:
        return "backward"
    elif task_name in realtime_tasks:
        return "realtime"
    elif task_name in forward_tasks:
        return "forward"
    else:
        raise ValueError(f"Unknown task name: {task_name}")


def build_prompt(doc, index):
    task = doc["task"]
    if not is_forward_task(doc):
        question = doc["question"]
        options = doc["options"]
        formatted_options = "; ".join(f"{chr(65 + i)}. {option}" for i, option in enumerate(options)) + ";"
        prompt = BR_PROMPT_TEMPLATE.format(question, formatted_options)
    else:
        if task == "REC":
            activity = doc["activity"]
            question = "How many times did they " + activity + "?"
            prompt = REC_PROMPT_TEMPLATE.format(question)
        elif task == "SSR":
            step = doc["test_info"][index]["step"]
            prompt = SSR_PROMPT_TEMPLATE.format(step)
        elif task == "CRR":
            question = doc["question"]
            prompt = CRR_PROMPT_TEMPLATE.format(question)
    return prompt


# can't be used, because multiround generation for chat models has not implemented yet
def ovo_doc_to_messages(doc, lmms_eval_specific_kwargs):
    pass


def ovo_back_real_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    text = build_prompt(doc, index=None)

    return text


# prepare for multiround generation
def ovo_forward_doc_to_text(doc, lmms_eval_specific_kwargs=None, previous_output=None, round_idx=None, previous_round_info=None):
    if round_idx is None:
        prompt = build_prompt(doc, index=0)
        return prompt
    else:
        i = round_idx
    terminal_sign = True if i == len(doc["test_info"]) else False
    if not terminal_sign:
        prompt = build_prompt(doc, index=i)
        lmms_eval_specific_kwargs["round_idx"] = i
        visuals = ovo_doc_to_visual(doc, lmms_eval_specific_kwargs)
        lmms_eval_specific_kwargs.pop("round_idx")
        return visuals, prompt, terminal_sign, previous_output, None
    else:
        return None, None, terminal_sign, previous_output, None


def ovo_doc_to_visual(doc, lmms_eval_specific_kwargs):
    assert lmms_eval_specific_kwargs["data_dir"] is not None
    if "round_idx" in lmms_eval_specific_kwargs:
        i = lmms_eval_specific_kwargs["round_idx"]
        chunk_video_path = os.path.join(lmms_eval_specific_kwargs["data_dir"], f'{doc["id"]}_{i}.mp4')
    elif is_forward_task(doc):
        # cause of logic of sending lmms_eval_specific_kwargs to doc_to_visual at initial round in multi round generation
        chunk_video_path = os.path.join(lmms_eval_specific_kwargs["data_dir"], f'{doc["id"]}_{0}.mp4')
    else:
        chunk_video_path = os.path.join(lmms_eval_specific_kwargs["data_dir"], f'{doc["id"]}.mp4')
    assert os.path.exists(chunk_video_path), f"Video chunk path does not exists:{chunk_video_path} !"

    return [chunk_video_path]


def ovo_doc_to_target(doc, lmms_eval_specific_kwargs=None):
    chr(65 + doc["gt"])


def ovo_back_real_process_results(doc, results):
    if isinstance(results, list) and isinstance(results[0], list):
        response = results[0][0].strip()
    else:
        response = results[0].strip()
    result = {"id": doc["id"], "task": doc["task"], "question": doc["question"], "response": response, "ground_truth": chr(65 + doc["gt"])}
    return {"back_real_acc": result}


def ovo_forward_process_results(doc, results):
    if isinstance(results, list) and isinstance(results[0], list):
        results = results[0][0]
    else:
        results = results[0]
    for i in range(len(doc["test_info"])):
        doc["test_info"][i]["response"] = results[i]
    return {"forward_acc": doc}


def ovo_back_real_acc(results, args):
    results, scores = calculate_score_backward_realtime(results)
    task_name = get_task_type(list(scores.keys())[0])
    path = generate_submission_file(f"{task_name}_acc_results.json", args)
    with open(path, "w") as f:
        json.dump(results, f, indent=4)

    scores_summary_dict = {}
    avg_scores = []
    for k, v in scores.items():
        scores_summary_dict[k] = 100 * sum(v) / len(v)
        avg_scores.append(sum(v) / len(v))

    total_avg_score = 100 * sum(avg_scores) / len(avg_scores)
    scores_summary_dict["average"] = total_avg_score
    path = generate_submission_file(f"{task_name}_acc_scores.json", args)

    print(scores_summary_dict)
    with open(path, "w") as f:
        json.dump(scores_summary_dict, f, indent=4)

    return scores_summary_dict["average"]


def ovo_forward_acc(results, args):
    results, scores = calculate_score_forward(results)
    path = generate_submission_file("forward_acc_results.json", args)
    with open(path, "w") as f:
        json.dump(results, f, indent=4)

    scores_summary_dict = {}
    avg_scores = []
    for k, v in scores.items():
        scores_summary_dict[k] = 100 * sum(v) / len(v)
        avg_scores.append(sum(v) / len(v))

    total_avg_score = 100 * sum(avg_scores) / len(avg_scores)
    scores_summary_dict["average"] = total_avg_score
    path = generate_submission_file("forward_acc_scores.json", args)

    print(scores_summary_dict)
    with open(path, "w") as f:
        json.dump(scores_summary_dict, f, indent=4)

    return scores_summary_dict["average"]
