import os
import yaml
import logging
import random
import pandas as pd

from pathlib import Path

eval_logger = logging.getLogger("lmms-eval")

try:
    from pywsd.utils import lemmatize_sentence
except ImportError:
    eval_logger.debug("pywsd not installed. Please install pywsd to use this module.")
    eval_logger.debug("You can install it by running 'pip install pywsd'")

from lmms_eval.tasks._task_utils.video_loader import get_cache_dir, get_video
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import numpy as np


OPTIONS = ["A", "B", "C", "D", "E"]


with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

stopwords = set(pd.read_csv(Path(__file__).parent / "stopwords.csv").squeeze())

cache_dir = get_cache_dir(config, "NExTVideo")


def nextqa_doc_to_visual(doc):
    return [get_video(cache_dir, doc["video"])]


def nextqa_doc_to_text(doc, model_specific_prompt_kwargs=None):
    question = doc["question"].strip()
    if "pre_prompt" in model_specific_prompt_kwargs and model_specific_prompt_kwargs["pre_prompt"] != "":
        question = f"{model_specific_prompt_kwargs['pre_prompt']}{question}"
    if "post_prompt" in model_specific_prompt_kwargs and model_specific_prompt_kwargs["post_prompt"] != "":
        question = f"{question}{model_specific_prompt_kwargs['post_prompt']}"
    return question


def nextqa_doc_to_text_mc(doc, model_specific_prompt_kwargs=None):
    if model_specific_prompt_kwargs is None:
        model_specific_prompt_kwargs = {}
    question = [doc["question"].strip()]
    for i in range(5):
        question.append(f"{OPTIONS[i]}. {doc[f'a{i}'].strip()}")
    question = "\n".join(question)
    if "pre_prompt" in model_specific_prompt_kwargs and model_specific_prompt_kwargs["pre_prompt"] != "":
        question = f"{model_specific_prompt_kwargs['pre_prompt']}{question}"
    if "post_prompt" in model_specific_prompt_kwargs and model_specific_prompt_kwargs["post_prompt"] != "":
        question = f"{question}{model_specific_prompt_kwargs['post_prompt']}"
    return question


def nextqa_mc_process_results(doc, results):
    pred = results[0]
    index2ans, all_choices = get_multi_choice_info(doc)
    parsed_pred = parse_multi_choice_response(pred, all_choices, index2ans)
    return {
        "exact_match": parsed_pred == OPTIONS[doc["answer"]],
    }


def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L10
    """
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f"{choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A. B. C. D.
            if f"{choice}." in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)  # -1 will be ignored anyway
                # start_indexes = [generated_response.index(f'({can})') for can in candidates]
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index


def nextqa_doc_to_target(doc):
    return doc["answer"]


def remove_stop(sentence):
    sentence.replace("</s>", "")  # video-llava
    words = lemmatize_sentence(sentence)
    words = [w for w in words if not w in stopwords]
    return " ".join(words)


def get_multi_choice_info(doc):
    all_choices = []
    index2ans = {}
    for i in range(5):
        index2ans[OPTIONS[i]] = doc[f"a{i}"].strip()
        all_choices.append(OPTIONS[i])

    return index2ans, all_choices


####################### WUPS ################################
# The following code copied from                            #
# https://github.com/doc-doc/NExT-OE/blob/main/metrics.py   #
#############################################################

# ====================================================
# @Time    : 13/9/20 4:19 PM
# @Author  : Xiao Junbin
# @Email   : junbin@comp.nus.edu.sg
# @File    : metrics.py
# ====================================================


def wup(word1, word2, alpha):
    """
    calculate the wup similarity
    :param word1:
    :param word2:
    :param alpha:
    :return:
    """
    # print(word1, word2)
    if word1 == word2:
        return 1.0

    w1 = wordnet.synsets(word1)
    w1_len = len(w1)
    if w1_len == 0:
        return 0.0
    w2 = wordnet.synsets(word2)
    w2_len = len(w2)
    if w2_len == 0:
        return 0.0

    # match the first
    word_sim = w1[0].wup_similarity(w2[0])
    if word_sim is None:
        word_sim = 0.0

    if word_sim < alpha:
        word_sim = 0.1 * word_sim
    return word_sim


def wups(words1, words2, alpha):
    """

    :param pred:
    :param truth:
    :param alpha:
    :return:
    """
    sim = 1.0
    flag = False
    for w1 in words1:
        max_sim = 0
        for w2 in words2:
            word_sim = wup(w1, w2, alpha)
            if word_sim > max_sim:
                max_sim = word_sim
        if max_sim == 0:
            continue
        sim *= max_sim
        flag = True
    if not flag:
        sim = 0.0
    return sim


def get_wups(pred, truth, alpha):
    """
    calculate the wups score
    :param pred:
    :param truth:
    :return:
    """
    pred = word_tokenize(pred)
    truth = word_tokenize(truth)
    item1 = wups(pred, truth, alpha)
    item2 = wups(truth, pred, alpha)
    value = min(item1, item2)
    return value


################ END WUPS ################################


def nextqa_process_results(doc, results):
    pred = results[0]
    answer = doc["answer"]
    pred_ans = remove_stop(pred)
    gt_ans = remove_stop(answer)
    qtype = doc["type"]
    if qtype == "TP":
        qtype = "TN"
    if qtype == "DC" or qtype == "DB":
        cur_0 = 1 if pred_ans == gt_ans else 0
        cur_9 = cur_0
    else:
        cur_0 = get_wups(pred_ans, gt_ans, 0)
        cur_9 = get_wups(pred_ans, gt_ans, 0.9)
    return {"WUPS": {"0": cur_0, "0.9": cur_9, "qtype": qtype}}


def nextqa_aggregate_results(results):
    qtypes = ["CW", "CH", "TN", "TC", "DB", "DC", "DL", "DO"]
    num = {"CW": 0, "CH": 0, "TN": 0, "TC": 0, "DB": 0, "DC": 0, "DL": 0, "DO": 0}
    over_num = {"C": 0, "T": 0, "D": 0}
    wups0 = {"CW": 0, "CH": 0, "TN": 0, "TC": 0, "DB": 0, "DC": 0, "DL": 0, "DO": 0}
    wups9 = {"CW": 0, "CH": 0, "TN": 0, "TC": 0, "DB": 0, "DC": 0, "DL": 0, "DO": 0}
    ref_num = 0
    for result in results:
        qtype = result["qtype"]
        num[qtype] += 1
        over_num[qtype[0]] += 1
        ref_num += 1
        cur_0 = result["0"]
        cur_9 = result["0.9"]
        wups0[qtype] += cur_0
        wups9[qtype] += cur_9

    wups0_all = wups9_all = 0
    wups0_e = wups0_t = wups0_c = 0
    for qtype in qtypes:
        wups0_all += wups0[qtype]
        wups9_all += wups9[qtype]
        if qtype[0] == "C":
            wups0_e += wups0[qtype]
        if qtype[0] == "T":
            wups0_t += wups0[qtype]
        if qtype[0] == "D":
            wups0_c += wups0[qtype]

        if num[qtype] != 0:
            wups0[qtype] = wups0[qtype] / num[qtype]
            wups9[qtype] = wups9[qtype] / num[qtype]
        else:
            wups0[qtype] = 0
            wups9[qtype] = 0

    # num_e = over_num["C"]
    # num_t = over_num["T"]
    # num_c = over_num["D"]

    # wups0_e /= num_e
    # wups0_t /= num_t
    # wups0_c /= num_c

    wups0_all /= ref_num
    wups9_all /= ref_num

    for k in qtypes:
        wups0[k] = wups0[k] * 100
        wups9[k] = wups9[k] * 100

    # wups0_e *= 100
    # wups0_t *= 100
    # wups0_c *= 100
    wups0_all *= 100

    return wups0_all
