import os
import json
from pycocoevalcap.eval import COCOEvalCap, Bleu, Meteor, Rouge, Cider, Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocotools.coco import COCO
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from pathlib import Path

import yaml
import sys

from loguru import logger as eval_logger

dir_name = os.path.dirname(os.path.abspath(__file__))

VATEX_METRICS = ["Bleu_4", "Bleu_3", "Bleu_2", "Bleu_1", "METEOR", "ROUGE_L", "CIDEr"]  # , "SPICE"]

# with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
#     raw_data = f.readlines()
#     safe_data = []
#     for i, line in enumerate(raw_data):
#         # remove function definition since yaml load cannot handle it
#         if "!function" not in line:
#             safe_data.append(line)

#     config = yaml.safe_load("".join(safe_data))

hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
# cache_dir = os.path.join(hf_home, cache_dir)
# base_cache_dir = config["dataset_kwargs"]["cache_dir"]
base_cache_dir = os.path.expanduser(hf_home)


def vatex_ZH_doc_to_visual(doc):
    with open(Path(__file__).parent / "vatex_val_zh.yaml", "r") as f:
        raw_data = f.readlines()
        safe_data = []
        for i, line in enumerate(raw_data):
            # remove function definition since yaml load cannot handle it
            if "!function" not in line:
                safe_data.append(line)
    cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = doc["videoID"] + ".mp4"
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    elif os.path.exists(video_path.replace("mp4", "MP4")):
        video_path = video_path.replace("mp4", "MP4")
    elif os.path.exists(video_path.replace("mp4", "mkv")):
        video_path = video_path.replace("mp4", "mkv")
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


def vatex_test_doc_to_visual(doc):
    with open(Path(__file__).parent / "vatex_test.yaml", "r") as f:
        raw_data = f.readlines()
        safe_data = []
        for i, line in enumerate(raw_data):
            # remove function definition since yaml load cannot handle it
            if "!function" not in line:
                safe_data.append(line)
    cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = doc["videoID"] + ".mp4"
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    elif os.path.exists(video_path.replace("mp4", "MP4")):
        video_path = video_path.replace("mp4", "MP4")
    elif os.path.exists(video_path.replace("mp4", "mkv")):
        video_path = video_path.replace("mp4", "mkv")
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


def vatex_ZH_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    few_shot_prompt = """[视频1] 输出:一个穿黑运动服、戴红色头盔的男人正在攀登雪山。\n[视频2] 输出:一个戴着耳机男人在电脑面前模拟打架子鼓。\n[视频3] 输出:一个穿黑色短袖的男子的男子，双手十指交叉放在胸前，肘部放在面前的桌子上，桌子上有一台电脑，不一会儿，男子半个手臂都放在了桌子上。\n[视频4] 输出:一位女士在她的手上涂抹少量的面霜，并且在她的眼睛下涂抹。\n"""
    return lmms_eval_specific_kwargs["prompt"] + "\n" + few_shot_prompt


def vatex_test_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    few_shot_prompt = """[video1] output: A man picks up a can of shoe paste, a towel, and brush from a table.\n[video2] output: A person places the frying pan on the stove and then another person flips over the food that is in it.\n[video3] output: A woman describes and demonstrates how to create a colorful cross stitch design.\n[video4] output: A little girl uses the grass in her yard as well as a blue mat to do flips.\n"""
    return lmms_eval_specific_kwargs["prompt"] + "\n" + few_shot_prompt


def vatex_process_result(doc, result):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name, value: metric value
    """
    pred = result[0] if len(result) > 0 else ""

    data_dict = {"answer": doc["enCap"], "pred": pred, "video_id": doc["videoID"]}

    return {f"vatex_{metric}": data_dict for metric in VATEX_METRICS}


def vatex_process_CN_result(doc, result):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name, value: metric value
    """
    pred = result[0] if len(result) > 0 else ""

    data_dict = {"answer": doc["chCap"], "pred": pred, "video_id": doc["videoID"]}

    return {f"vatex_{metric}": data_dict for metric in VATEX_METRICS}


def vatex_aggregation_result(results, metric, args=None):
    scorers = [(Bleu(4), "Bleu_1"), (Bleu(4), "Bleu_2"), (Bleu(4), "Bleu_3"), (Bleu(4), "Bleu_4"), (Meteor(), "METEOR"), (Rouge(), "ROUGE_L"), (Cider(), "CIDEr")]  # , (Spice(), "SPICE")]
    scorers_dict = {s[1]: s for s in scorers}

    stored_results = []
    # In order to make the coco eval tools to successfully create index
    # We need at least two dict in the dataset
    # 'annotation' and 'images'
    # 'annotation' exactly reproduce the original annotation
    # 'images' however only need the image id which is contained in the file name
    dataset = {"annotations": [], "images": []}
    idx = 0
    for result in results:
        stored_results.append({"image_id": result["video_id"], "caption": result["pred"]})
        for a in result["answer"]:
            dataset["annotations"].append({"image_id": result["video_id"], "caption": a, "id": idx})
            idx += 1
        dataset["images"].append({"id": result["video_id"]})

    coco = COCO()
    # Manually create index here
    coco.dataset = dataset
    coco.createIndex()

    vatex_result = coco.loadRes(stored_results)
    vatex_eval = COCOEvalCap(coco, vatex_result)

    imgIds = vatex_eval.params["image_id"]
    gts = {}
    res = {}
    for imgId in imgIds:
        gts[imgId] = vatex_eval.coco.imgToAnns[imgId]
        res[imgId] = vatex_eval.cocoRes.imgToAnns[imgId]

    eval_logger.info("tokenization...")
    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    eval_logger.info(f"Computing {metric} scores...")

    score, scores = scorers_dict[metric][0].compute_score(gts, res)
    # When metric is one of the Bleu, score will be a list
    if type(score) == list:
        n = int(metric.split("_")[-1])
        score = score[n - 1]

    path = generate_submission_file("vatex_captions_val_results.json", args)

    with open(path, "w") as f:
        json.dump(stored_results, f, indent=4)

    eval_logger.info(f"Results saved to {path}")

    return score


def vatex_bleu4(results, args=None):
    return vatex_aggregation_result(results, "Bleu_4", args)


def vatex_bleu3(results, args=None):
    return vatex_aggregation_result(results, "Bleu_3", args)


def vatex_bleu2(results, args=None):
    return vatex_aggregation_result(results, "Bleu_2", args)


def vatex_bleu1(results, args=None):
    return vatex_aggregation_result(results, "Bleu_1", args)


def vatex_meteor(results, args=None):
    return vatex_aggregation_result(results, "METEOR", args)


def vatex_rougel(results, args=None):
    return vatex_aggregation_result(results, "ROUGE_L", args)


def vatex_cider(results, args=None):
    return vatex_aggregation_result(results, "CIDEr", args)


def vatex_spice(results, args=None):
    return vatex_aggregation_result(results, "SPICE", args)


def vatex_test_process_result(doc, result):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case vatex_passthrough), value: metric value
    """
    return {"vatex_passthrough": {"pred": result, "image_id": doc["image_id"]}}


def vatex_test_aggregation_result(results, args):
    stored_results = []
    for result in results:
        stored_results.append({"image_id": result["image_id"], "caption": result["pred"]})

    path = generate_submission_file("vatex_captions_test2014_alg_results.json", args)
    eval_logger.info("Storing prediction that can be submitted to the server ...")
    with open(path, "w") as f:
        json.dump(stored_results, f, indent=4)

    eval_logger.info(f"Your test result has been stored into {path}. Make sure you also have the val result stored to submit to the server on https://codalab.lisn.upsaclay.fr/competitions/7404#participate.")
