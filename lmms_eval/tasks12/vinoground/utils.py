import json
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


textscore_dict, videoscore_dict = {}, {}


def prep_data():
    global textscore_dict, videoscore_dict
    cache_dir = os.path.join(base_cache_dir, cache_name)
    with open(os.path.join(cache_dir, "vinoground_videos", "vinoground_textscore.json")) as f:
        textscore_list = json.load(f)
    textscore_dict = {}
    for item in textscore_list:
        textscore_dict[item["idx"]] = item
    with open(os.path.join(cache_dir, "vinoground_videos_concated", "vinoground_videoscore.json")) as f:
        videoscore_list = json.load(f)
    videoscore_dict = {}
    for item in videoscore_list:
        videoscore_dict[item["idx"]] = item
    return textscore_dict, videoscore_dict


def vinoground_doc_to_visual(doc):
    if len(textscore_dict) == 0:
        prep_data()
    cache_dir = os.path.join(base_cache_dir, cache_name)
    idx, question_type = "_".join(doc["index"].split("_")[:2]), doc["index"].split("_")[2]
    scoredict = textscore_dict if question_type == "text" else videoscore_dict

    video_path = os.path.join(cache_dir, scoredict[idx]["video_name"])
    if not os.path.exists(video_path):
        raise Exception(f"video path:{video_path} does not exist, please check")
    return [video_path]


def vinoground_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if len(textscore_dict) == 0:
        prep_data()
    idx, question_type = "_".join(doc["index"].split("_")[:2]), doc["index"].split("_")[2]
    scoredict = textscore_dict if question_type == "text" else videoscore_dict

    return scoredict[idx]["question"] + "\nPlease only output one English character."


def vinoground_process_results(doc, results):
    pred = results[0]

    major = doc["major"]
    minors = doc["minor"]
    categories = ["all", major]
    if minors is not None:
        categories.extend(minors.split(";"))
    idx, question_type = "_".join(doc["index"].split("_")[:2]), doc["index"].split("_")[2]
    data_dict = {"index": idx, "categories": categories, "question_type": question_type, "pred": pred}

    return {"vinoground_score": data_dict}


def vinoground_aggregate_results(results):
    matrix = np.zeros((500, 7), dtype=np.int8)
    textscore_dict, videoscore_dict = prep_data()

    category_all = {}
    category_text = {}
    category_video = {}
    category_group = {}
    index_to_categories = {}

    for result in results:
        idx, categories, question_type, pred = result["index"], result["categories"], result["question_type"], result["pred"]
        matrix_col = 0 if "pos" in idx else 1
        if question_type == "video":
            matrix_col += 3
        if question_type == "text":
            gt = textscore_dict[idx]["GT"]
        else:
            gt = videoscore_dict[idx]["GT"]
        idx = int(idx.split("_")[0])
        matrix[idx, matrix_col] = pred[0].lower() == gt.lower()

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
