import ast
import os
import cv2
import numpy as np
import zipfile
import yaml
from collections import defaultdict
from pathlib import Path
from PIL import Image
from typing import Any, Optional
from huggingface_hub import hf_hub_download

with open(Path(__file__).parent / "lemonade.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for line in raw_data:
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

HF_HOME = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(HF_HOME)
cache_dir = config["dataset_kwargs"]["cache_dir"]
videos_dir = os.path.join(base_cache_dir, cache_dir)

max_num_frames = config.get("lmms_eval_specific_kwargs", {}).get("max_num_frames", 8)


def load_video(video_file: str, start_frame: int, end_frame: int, max_num_frames: int = max_num_frames) -> list[Image.Image]:
    """
    Args:
        video_file: Path to the video file.
        start_frame: Starting frame index.
        end_frame: Ending frame index.
        max_num_frames: Number of frames to sample from the video segment.
    Returns:
        List of PIL Image objects representing sampled frames
    """

    cap = cv2.VideoCapture(video_file)
    try: 
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = max(0, start_frame)
        end_frame = min(end_frame, total_frames - 1)
        total_valid_frames = end_frame - start_frame + 1
        num_frames = min(max_num_frames, total_valid_frames)
        step = total_valid_frames / num_frames
        frame_indices = [int(start_frame + i * step) for i in range(num_frames)]
        frames = []
        for target_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            success, frame = cap.read()
            if not success:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb).convert("RGB")
            frames.append(pil_img)

        return frames
    finally:
        cap.release()

def parse_options(options: list[str]) -> str:
    """
    Format a list of multiple-choice options into a string.
    The function assigns letters to each option and returns them in a newline-separated string.    
    
    Args:
        options (list[str]): A list of option strings.

    Returns:
        str: A formatted string with each option on a new line, prefixed by its corresponding letter.
    """

    option_letters = [chr(ord("A") + i) for i in range(len(options))]

    if all(option.startswith(f"{letter}.") for option, letter in zip(options, option_letters)):
        return "\n".join(options)

    choices_str = "\n".join([f"{option_letter}. {option}" for option_letter, option in zip(option_letters, options)])
    return choices_str


def lemonade_doc_to_visual(doc: dict[str, Any]) -> list[Image.Image]:
    """
    Load video frames for a given entry in the LEMONADE dataset.

    Args:
        doc: A dictionary representing an entry in the dataset.
    Returns:
        frames: List of PIL Image objects representing sampled frames
    """

    video_filename = doc["Clip"] + "_hololens.mp4"
    video_path = os.path.join(videos_dir, video_filename)

    if os.path.exists(video_path):
        start = int(doc["Start"])
        end = int(doc["End"])
        frames = load_video(video_path, start, end, max_num_frames=max_num_frames)
    else:
        raise FileNotFoundError(
            f"Video file not found: {video_path}. "
            f"Expected video for clip '{doc['Clip']}' at {video_path}"
        )
    return frames
    

def lemonade_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs: Optional[dict[str, Any]] = None) -> str:
    """
    Convert a LEMONADE dataset entry into a formatted text prompt.
    Args:
        doc: A dictionary representing an entry in the dataset.
        lmms_eval_specific_kwargs: Optional dictionary for additional prompt formatting.
    Returns:
        str: A formatted prompt string ready for model input
    """

    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
        
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    
    question = "Question: " + doc["Question"]
    parsed_options = parse_options(ast.literal_eval(doc["Answers"]))
    choices = "Choices:\n" + parsed_options

    return f"{pre_prompt}{question}\n{choices}{post_prompt}"


def get_multi_choice_info(options: list[str]) -> tuple[dict[str, str], list[str]]:
    """
    Map a list of options to letter labels (A, B, C, ...).
    
    Args:
        options: The set of answer options
    Returns:
        tuple[dict[str, str], list[str]]: 
            - index2ans: Mapping from letters to option text.
            - all_choices: List of the assigned letters.
    """
    
    if not isinstance(options, list):
        raise TypeError(f"Expected list of options, got {type(options)}: {options}")
   
    start_chr = "A"
    all_choices = []
    index2ans = {}
    for i, option in enumerate(options):
        index2ans[chr(ord(start_chr) + i)] = option
        all_choices.append(chr(ord(start_chr) + i))

    return index2ans, all_choices


def parse_multi_choice_response(response: str, all_choices: list[str], index2ans: dict[str, str]) -> str:
    """
    Parse a model response and return the predicted choice label (e.g., "A", "B", "C", "D"). 

    Args:
        response (str): The generated response to parse.
        all_choices (list[str]): The set of valid choice labels.
        index2ans (dict[str, str]): Mapping from choice labels to their full answer text.
    Returns:
        str: The predicted choice label.
    """

    if response == "API Error":
        return "API Error"

    if response == "":
        return "Empty Response"

    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " " 

    index_ans = True
    ans_with_brack = False
    ans_with_period = False
    ans_with_colon = False
    candidates = []

    for choice in all_choices:
        if f"{choice}." in response:
            candidates.append(choice)
            ans_with_period = True
    for choice in all_choices: 
        if f"{choice}:" in response:
            candidates.append(choice)
            ans_with_colon = True
    if len(candidates) == 0:
        for choice in all_choices:
            if f"({choice})" in response:
                candidates.append(choice)
                ans_with_brack = True
    if len(candidates) == 0:
        for choice in all_choices: 
            if f"{choice} " in response:
                candidates.append(choice)
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False 
    if len(candidates) == 0:
        pred_index = "A"

    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_period:
                for can in candidates:
                    index = response.rfind(f"{can}.")
                    start_indexes.append(index)
            elif ans_with_colon:
                for can in candidates:
                    index = response.rfind(f"{can}:")
                    start_indexes.append(index)
            elif ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        pred_index = candidates[np.argmax(start_indexes)]
    else:
        pred_index = candidates[0]

    return pred_index


def lemonade_process_results(doc: dict[str, Any], results: list[Any]) -> dict[str, dict]:
    """
    Process the results from the model and compute accuracy.
    
    Args:
        doc: A dictionary representing an entry in the dataset.
        results: List of model outputs.
    Returns:
        A dictionary containing accuracy information. 
    """
    
    pred = results[0]
    index2ans, all_choices = get_multi_choice_info(ast.literal_eval(doc["Answers"]))
    parsed_pred = parse_multi_choice_response(pred, all_choices, index2ans)

    acc = {
        "QID": doc["QID"],
        "category": doc["Category"],
        "subcategory": doc["Subcategory"],
        "difficulty": doc["Difficulty"],
        "answer": doc["Correct Answer"],
        "parsed_pred": parsed_pred,
        "original_pred": pred
    }    
    return {"acc": acc}


def lemonade_aggregate_results(results: list[dict[str, Any]]) -> float:
    """
    Aggregate the results from the evaluation.
    
    Args:
        results: List of dicts containing individual evaluation results.
    Returns:
        overall_acc: Overall accuracy.

    """
    def compute_accuracy(grouped_results):
        acc_dict = {}
        for key, samples in grouped_results.items():
            correct = sum([r["parsed_pred"] == r["answer"] for r in samples])
            total = len(samples)
            acc = round(correct / total, 5) if total > 0 else 0.0
            stderr = round(np.sqrt(acc * (1 - acc) / total), 5) if total > 0 else 0.0
            acc_dict[key] = {
                "num": total,
                "acc": acc,
                "acc_stderr": stderr,
            }
        return acc_dict

    qid_results = defaultdict(list)
    category_results = defaultdict(list)
    subcategory_results = defaultdict(list)
    difficulty_results = defaultdict(list)

    valid_results = [r for r in results if r["parsed_pred"] != "API Error"]

    for r in valid_results:
        qid_results[r["QID"]].append(r)
        category_results[r["category"]].append(r)
        subcategory_results[r["subcategory"]].append(r)
        difficulty_results[r["difficulty"]].append(r)

    qid_acc = compute_accuracy(qid_results)
    category_acc = compute_accuracy(category_results)
    subcategory_acc = compute_accuracy(subcategory_results)
    difficulty_acc = compute_accuracy(difficulty_results)

    total_correct = sum([r["parsed_pred"] == r["answer"] for r in valid_results])
    total = len(valid_results)
    overall_acc = round(total_correct / total, 5) if total > 0 else 0.0
    overall_stderr = round(np.sqrt(overall_acc * (1 - overall_acc) / total), 5) if total > 0 else 0.0

    print("\nResults:")

    print("\nAccuracy per QID:")
    for k, v in qid_acc.items():
        print(f"  {k}: {v['acc']} ± {v['acc_stderr']} ({v['num']} examples)")

    print("\nAccuracy per Category:")
    for k, v in category_acc.items():
        print(f"  {k}: {v['acc']} ± {v['acc_stderr']} ({v['num']} examples)")

    print("\nAccuracy per Subcategory:")
    for k, v in subcategory_acc.items():
        print(f"  {k}: {v['acc']} ± {v['acc_stderr']} ({v['num']} examples)")

    print("\nAccuracy per Difficulty:")
    for k, v in difficulty_acc.items():
        print(f"  {k}: {v['acc']} ± {v['acc_stderr']} ({v['num']} examples)")

    print(f"\nOverall Accuracy: {overall_acc} ± {overall_stderr} ({total} examples)")

    return overall_acc
