import os

import pandas as pd
from lmms_eval.tasks.plm_videobench.eval_utils import *

# Load default config parameters
config = load_defualt_config()

# Load video paths
video_base_dir = config["plm_fgqa"]["video_base_dir"]
assert video_base_dir is not None, f"video_base_dir is not set. Please double check if you have downloaded the videos and set the correct path in _default_template_yaml."

# Load the number of video frames
num_video_frames = config["plm_stc"]["num_video_frames"]
assert num_video_frames is not None, f"num_video_frames must not be None."


def qa_template(entry):

    question = f"Question: {entry['question']}\n"
    question += "Options:\n"
    answer = entry["answer"]
    answer_idx = -1
    for idx, c in entry["options"].items():
        idx = int(idx.split("_")[-1])
        question += f"({chr(ord('A') + idx)}) {c}\n"
        if c == answer:
            answer_idx = idx
    assert answer_idx != -1, "Answer not found"
    question = question.rstrip()
    answer = f"({chr(ord('A') + answer_idx)}) {answer}"

    # Add the instruction to question
    question_prompt = "\nOnly give the best option."  # to change
    question += question_prompt

    return question, answer


def plm_fgqa_doc_to_visual(doc):
    video_id = doc["video"]
    video_path = os.path.join(video_base_dir, video_id)

    return [video_path]


def plm_fgqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    msg = "'prompt' must be specified in lmms_eval_specific_kwargs for the 'plm_fgqa' task."
    assert lmms_eval_specific_kwargs and "prompt" in lmms_eval_specific_kwargs, msg

    question, _ = qa_template(doc)
    prompt_template = lmms_eval_specific_kwargs["prompt"]
    prompt = prompt_template.format(question=question, answer="{answer}")

    return prompt


def plm_fgqa_process_results(doc, results):
    pred = results[0]

    prompt, answer = qa_template(doc)
    accuracy = float(check_ans(pred=pred, gt=answer))

    results_dict = {kk: vv for kk, vv in doc.items() if not kk == "metadata"}

    results_dict["prompt"] = prompt
    results_dict["prediction"] = pred
    results_dict["accuracy"] = accuracy

    return {"plm_fgqa_scores": results_dict}


def plm_fgqa_aggregate_results(results):
    df_res = pd.DataFrame(results)

    # -- add the onevsall accuracy for all q_kinds
    multibinary_accuracy = df_res.groupby("qa_uid").accuracy.apply(lambda x: all([i for i in x])).mean().item()

    printable_results = {"multibinary_accuracy": multibinary_accuracy, "num_instances": len(results)}

    return printable_results


def check_ans(pred: str, gt: str) -> bool:
    """
    Check if the predicted answer matches the ground truth answer.

    Args:
        pred (str): The predicted answer.
        gt (str): The ground truth answer.

    Returns:
        bool: True if the predicted answer matches the ground truth answer, False otherwise.
    """
    if "answer:" in pred.lower():
        pred = pred.lower().replace("answer:", "").strip()

    flag = False

    pred_list = pred.lower().split(" ")
    pred_option, pred_content = pred_list[0], " ".join(pred_list[0:])
    if len(pred_option) == 0:
        return False
    gt_list = gt.lower().split(" ")
    gt_option, gt_content = gt_list[0], " ".join(gt_list[1:])
    if gt_content[-1] == ".":
        gt_content = gt_content[:-1]
    if pred_option.replace(".", "") in gt_option:
        flag = True
    elif gt_option in pred_option:
        flag = True

    return flag
