import re
import os
import sys
from loguru import logger as eval_logger
import datetime
import lmms_eval.tasks._task_utils.file_utils as file_utils
from lmms_eval.filters.extraction import ExtendedRegexFilter
import json
import yaml
import torch
from pathlib import Path
import requests
import time
from pycocoevalcap.eval import COCOEvalCap, Bleu, Meteor, Rouge, Cider, Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
import numpy as np
from tqdm import tqdm

# import nltk
# nltk.download('punkt')
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from nltk.tokenize import word_tokenize
# from rouge import Rouge


with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))


# Different Implementations of BLEU-4 and Rouge with nltk and pycocoevalcap

# Use nltk to calculate BLEU-4 score
# def calculate_bleu4(reference, hypothesis):
#     smooth_fn = SmoothingFunction().method1
#     ref_tokens = word_tokenize(reference)
#     hyp_tokens = word_tokenize(hypothesis)
#     blue4_score = sentence_bleu([ref_tokens],
#                                 hyp_tokens,
#                                 weights=(0.25, 0.25, 0.25, 0.25),
#                                 smoothing_function=smooth_fn)
#
#     return blue4_score

# use pycocoevalcap to calculate BLEU-4 score
def calculate_bleu4(reference, hypothesis):
    ref = {0: [{'caption': reference}]}
    hyp = {0: [{'caption': hypothesis}]}

    tokenizer = PTBTokenizer()
    ref_tokenized = tokenizer.tokenize(ref)
    hyp_tokenized = tokenizer.tokenize(hyp)

    scorer = Bleu(4)
    score, _ = scorer.compute_score(ref_tokenized, hyp_tokenized)
    bleu4_score = score[3]
    return bleu4_score


# Use rouge to calculate ROUGE-L score
# def calculate_rouge(reference, hypothesis):
#     rouge = Rouge()
#     rouge_score = rouge.get_scores(hypothesis, reference, avg=True)
#     return rouge_score['rouge-l']['f']

# use pycocoevalcap to calculate ROUGE-L score
def calculate_rouge(reference, hypothesis):
    ref = {0: [{'caption': reference}]}
    hyp = {0: [{'caption': hypothesis}]}


    tokenizer = PTBTokenizer()
    ref_tokenized = tokenizer.tokenize(ref)
    hyp_tokenized = tokenizer.tokenize(hyp)
    # eval_logger.info("computing ROUGE score")
    scorer = Rouge()
    score, _ = scorer.compute_score(ref_tokenized, hyp_tokenized)

    return score

# A bit ugly here
# But the idea is that we will unzip all the zip files
# To HF HOME cache dir
# And load it here
HF_HOME = os.getenv("HF_HOME", "~/.cache/huggingface/")
cache_dir = config["dataset_kwargs"]["cache_dir"]
cache_dir = os.path.join(HF_HOME, cache_dir)
cache_dir = os.path.join(cache_dir, "videos")

from loguru import logger as eval_logger


# Pass in video path here
# Can only work correctly with video llm
def cuva_doc_to_visual(doc):
    video_path = doc["video_name"]
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    elif os.path.exists(video_path.replace("mp4", "MP4")):
        video_path = video_path.replace("mp4", "MP4")
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


# This is the place where you format your question
def cuva_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    questions= {
        "Description": "Watch the video and describe any anomaly events you see in the order they happen. Focus on what is different from normal, like who or what is involved and their actions.",
        "Cause": "Explain why the anomaly in the video are happening. Use what you see in the video to make logical reasoning about the root reasons behind these anomalies.Please ensure that your response is logically rigorous and directly related to the abnormal events in the video and the potential reasons behind them.",
        "Result": "Figure out what results and effect these anomalies have. Link the anomaly directly to their outcomes, like how they affect people or the environment. Your answer should be as clear and specific as possible, avoiding generalities and focusing directly on the video rather than summarizing the impact of a type of event on society."
    }
    question = questions[doc["task"]]
    return question


def cuva_doc_to_answer(doc):
    return doc["answer"]


# An example of showing how to custom metric
# Your metric name should have the same key name in your return dict
def cuva_process_results(doc, result):
    pred = result[0]
    return {
        "cuva_BLEU": {"pred": pred, "answer": doc["answer"], "task": doc["task"]},
        "cuva_ROUGE": {"pred": pred, "answer": doc["answer"], "task": doc["task"]},
        "cuva_BLEURT": {"pred": pred, "answer": doc["answer"], "task": doc["task"]},
    }

def cuva_aggregate_results(results, metric, args):
    # TODO: Add more metrics here
    score_functions = {
        "BLEU": calculate_bleu4,
        "ROUGE": calculate_rouge
    }

    if metric not in score_functions:
        raise ValueError("Unsupported metric")

    scores_dict = {'Description': [], 'Cause': [], 'Result': []}
    eval_logger.info(f"Calculating {metric} score")
    for result in tqdm(results):
        gt = result["answer"]
        pred = result["pred"]
        task = result["task"]
        scores_dict[task].append(score_functions[metric](gt, pred))

    mean_scores = {task: np.mean(scores) if scores else 0 for task, scores in scores_dict.items()}
    eval_logger.info(f"{metric} score: {mean_scores}")

    mean_score = sum(mean_scores.values()) / len(mean_scores)
    now_date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    scores_file_name = f"cuva-{metric}-{now_date_time}.json"
    path = file_utils.generate_submission_file(scores_file_name, args, subpath="scores")
    with open(path, "w") as f:
        json.dump(mean_scores, f)
    eval_logger.info(f"{metric} score file saved to {path}")
    return mean_score

def cuva_aggregate_results_bleurt(results, args):
    bleurt_version = 'lucadiliello/BLEURT-20'
    eval_logger.info(f"Loading BLEURT model {bleurt_version}, you can change to the small version BLEURT-20-D12 in tasks/cuva/utils.py")
    config = BleurtConfig.from_pretrained(bleurt_version)
    model = BleurtForSequenceClassification.from_pretrained(bleurt_version)
    tokenizer = BleurtTokenizer.from_pretrained(bleurt_version)

    scores_dict = {'Description': [], 'Cause': [], 'Result': []}
    eval_logger.info(f"Calculating BLEURT score")
    for result in tqdm(results):
        gt = result["answer"]
        pred = result["pred"]
        task = result["task"]
        model.eval()
        with torch.no_grad():
            inputs = tokenizer([gt], [pred], padding='longest', return_tensors='pt')
            res = model(**inputs).logits.flatten().tolist()
        scores_dict[task].append(res[0])

    mean_scores = {task: np.mean(scores) if scores else 0 for task, scores in scores_dict.items()}
    eval_logger.info(f"BLEURT score: {mean_scores}")
    mean_score = sum(mean_scores.values()) / len(mean_scores)

    now_date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    scores_file_name = f"cuva-BLEURT-{now_date_time}.json"
    path = file_utils.generate_submission_file(scores_file_name, args, subpath="scores")
    with open(path, "w") as f:
        json.dump(mean_scores, f)
    eval_logger.info(f"BLEURT score file saved to {path}")
    return mean_score

def cuva_BLEU(results, args):
    return cuva_aggregate_results(results, "BLEU", args)

def cuva_ROUGE(results, args):
    return cuva_aggregate_results(results, "ROUGE", args)

def cuva_BLEURT(results, args):
    return cuva_aggregate_results_bleurt(results, args)



