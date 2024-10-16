import os
from pathlib import Path

import numpy as np
import yaml
from loguru import logger as eval_logger

hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "vinoground.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


def vinoground_doc_to_visual(doc):
    cache_dir = os.path.join(base_cache_dir, cache_name)

    if doc["index"].split("_")[2] == "text":
        video_path = os.path.join(cache_dir, "vinoground_videos", "_".join(doc["index"].split("_")[:2]) + ".mp4")
    else:
        video_path = os.path.join(cache_dir, "vinoground_videos_concated", doc["index"].split("_")[0] + ".mp4")
    if not os.path.exists(video_path):
        raise Exception(f"video path:{video_path} does not exist, please check")
    return [video_path]


def vinoground_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if doc["index"].split("_")[2] == "text":
        pre_prompt = "Which caption best describes this video?"
        option_a = "A. " + doc["pos_cap"]
        option_b = "B. " + doc["neg_cap"]
        post_prompt = "Answer with the option's letter from the given choices directly. Please only output 1 English character."
        full_prompt = pre_prompt + "\n" + option_a + "\n" + option_b + "\n" + post_prompt
    else:
        pos_neg = doc["index"].split("_")[1]
        caption_in_question = doc[f"{pos_neg}_cap"]
        pre_prompt = "Which video segment matches this caption? Note: The video contains two segments separated by a 2-second black frame."
        caption = f"Caption: {caption_in_question}"
        options = "A. The first fragment (before black frame)\nB. The second fragment (after black frame)"
        post_prompt = "Answer with the option's letter from the given choices directly. Please only output 1 English character."
        full_prompt = pre_prompt + "\n" + caption + "\n" + options + "\n" + post_prompt
    return full_prompt


def vinoground_process_results(doc, results):
    pred = results[0]

    major = doc["major"]
    minors = doc["minor"]
    categories = [major]
    if minors is not None:
        categories.extend(minors.split(";"))
    question_type = doc["index"].split("_")[2]
    data_dict = {"index": doc["index"], "categories": categories, "question_type": question_type, "pred": pred}

    return {"vinoground_score": data_dict}


def vinoground_aggregate_results(results):
    matrix = np.zeros((500, 7), dtype=np.int8)

    category_all = {}
    category_text = {}
    category_video = {}
    category_group = {}
    index_to_categories = {}

    for result in results:
        index, categories, question_type, pred = result["index"], result["categories"], result["question_type"], result["pred"]
        matrix_col = 0 if "pos" in index else 1
        if question_type == "video":
            matrix_col += 3
        gt = "A" if "pos" in index else "B"
        idx = int(index.split("_")[0])
        matrix[idx, matrix_col] = pred[0].lower() == gt.lower()

        categories.append("all")
        if idx not in index_to_categories.keys():
            index_to_categories[idx] = categories

    matrix[:, 2] = matrix[:, 0] & matrix[:, 1]
    matrix[:, 5] = matrix[:, 3] & matrix[:, 4]
    matrix[:, 6] = matrix[:, 2] & matrix[:, 5]

    for i in range(500):
        for category in index_to_categories[i]:
            if category not in category_all.keys():
                category_all[category] = 0
                category_text[category] = 0
                category_video[category] = 0
                category_group[category] = 0

            category_all[category] += 1
            category_text[category] += matrix[i, 2]
            category_video[category] += matrix[i, 5]
            category_group[category] += matrix[i, 6]

    loginfo = "Categorical results:\n"
    for category in category_all.keys():
        loginfo += (
            f"{category}: text: {category_text[category] / category_all[category] * 100:.2f}%, video: {category_video[category] / category_all[category] * 100:.2f}%, group: {category_group[category] / category_all[category] * 100:.2f}%\n"
        )
    eval_logger.info(loginfo)

    return matrix[:, 2].mean() * 100, matrix[:, 5].mean() * 100, matrix[:, 6].mean() * 100
