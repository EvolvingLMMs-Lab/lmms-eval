import datetime
import yaml
import json
import os

import evaluate
import numpy as np
import spacy
from nltk.util import ngrams
from spacy.cli import download

from pathlib import Path
from difflib import SequenceMatcher as SM
from functools import partial

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from loguru import logger as eval_logger

with open(Path(__file__).parent / "_default_template_vcr_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

# Download the English and Chinese models
if config["metadata"]["load_package"]:
    try:
        nlp_en = spacy.load("en_core_web_sm")
        nlp_zh = spacy.load("zh_core_web_sm")
        nlp = {"en": nlp_en, "zh": nlp_zh}
        rouge = evaluate.load("rouge")
    except Exception as e:
        eval_logger.debug(f"Failed to load spacy models: {e}")
        download("en_core_web_sm")
        nlp_en = spacy.load("en_core_web_sm")
        download("zh_core_web_sm")
        nlp_zh = spacy.load("zh_core_web_sm")
        eval_logger.debug("Spacy models not loaded due to load_package is False. Please set load_package to True in the config file to load them.")
else:
    nlp = {"en": None, "zh": None}
    rouge = None

aggregate_results_template = {
    "max_sim_val": 0,
    "precision": 0,
    "recall": 0,
    "f1": 0,
    "jaccard": 0,
    "rouge1": 0,
}


def fast_filter(answer_text):
    if "I can't" in answer_text:
        return True
    elif "I cannot" in answer_text:
        return True
    elif "sorry" in answer_text.lower():
        return True
    if "无法" in answer_text:
        return True
    elif "抱歉" in answer_text:
        return True
    else:
        return False


def vcr_doc_to_visual(doc):
    return [doc["stacked_image"].convert("RGB")]


def vcr_doc_to_text(doc, model_specific_prompt_kwargs=None):
    if "pre_prompt" in model_specific_prompt_kwargs:
        pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
    if "post_prompt" in model_specific_prompt_kwargs:
        post_prompt = model_specific_prompt_kwargs["post_prompt"]
    return f"{pre_prompt}{post_prompt}"


def tokenize(text, language):
    """
    Tokenize the text and return the tokens.

    Parameters:
    text (str): The text to tokenize.
    language (str): The language of the text.

    Returns:
    list: The list of tokens.
    """
    assert language in ["en", "zh"]
    nlp_lang = nlp[language]
    processed_text = nlp_lang(text)
    return [token.text for token in processed_text]


def vcr_process_results_single(crossed_text, result, language):
    """
    Args:
        doc: an instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case vcr score), value: metric value
    """

    assert language in ["en", "zh"], f"Language {language} is not supported."

    if fast_filter(result):
        return {
            "crossed_text": crossed_text,
            "max_sim_val": 0,
            "max_sim_string": "",
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "jaccard": 0,
            "rouge1": 0,
            "exact_match": 0,
        }
    tokens_result = tokenize(result, language)
    tokens_crossed_text = tokenize(crossed_text, language)

    splitter = " " if language == "en" else ""
    ngrams_ = ngrams(tokens_result, len(tokens_crossed_text))
    max_sim_val = 0
    max_sim_string = ""
    max_sim_ngram = []
    tokens_crossed_text_set = set(tokens_crossed_text)
    ngrams_hasjoint = [ngram for ngram in ngrams_ if not set(ngram).isdisjoint(tokens_crossed_text_set)]

    for ngram in ngrams_hasjoint:
        result_ngram = splitter.join(ngram)
        similarity = SM(None, result_ngram, crossed_text).ratio()
        if similarity > max_sim_val:
            max_sim_val = similarity
            max_sim_string = result_ngram
            max_sim_ngram = ngram

    # Evaluate
    if len(max_sim_ngram) == 0:
        return {
            "crossed_text": crossed_text,
            "max_sim_val": 0,
            "max_sim_string": "",
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "jaccard": 0,
            "rouge1": 0,
            "exact_match": 0,
        }
    pred_set = set(max_sim_ngram)
    ref_set = set(tokens_crossed_text)
    correct_tokens = pred_set.intersection(ref_set)
    len_correct_tokens = len(correct_tokens)

    precision = len_correct_tokens / len(pred_set)
    recall = len_correct_tokens / len(ref_set)
    if (precision + recall) == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    union = pred_set.union(ref_set)
    jaccard = len_correct_tokens / len(union) if len(union) > 0 else 0
    rouge_1 = rouge.compute(
        predictions=[max_sim_string],
        references=[crossed_text],
        tokenizer=partial(tokenize, language=language),
        rouge_types=["rouge1"],
    )["rouge1"]
    exact_match = float(list(max_sim_ngram) == list(tokens_crossed_text))
    out = {
        "crossed_text": crossed_text,
        "max_sim_string": max_sim_string,
        "max_sim_val": max_sim_val,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
        "rouge1": rouge_1,
        "exact_match": exact_match,
    }
    return out


def vcr_en_process_results(doc, results):
    """
    Args:
        doc: an instance of the eval dataset
        results: [pred], with length = 1
    Returns:
        a dictionary with key: metric name (in this case vcr score), value: metric value
    """
    output = {
        "max_sim_val": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "jaccard": [],
        "rouge1": [],
        "exact_match": [],
    }
    crossed_text = doc["crossed_text"]
    for i in range(len(crossed_text)):
        tmp = vcr_process_results_single(crossed_text[i], results[0], "en")
        for k in output.keys():
            output[k].append(
                {
                    "score": tmp[k],
                    "pred_ngram": tmp["max_sim_string"],
                    "gt_ngram": crossed_text[i],
                    "caption": doc["caption"],
                }
            )
    return output


def vcr_zh_process_results(doc, results):
    """
    Args:
        doc: an instance of the eval dataset
        results: [pred], with length = 1
    Returns:
        a dictionary with key: metric name (in this case vcr score), value: metric value and other info
    """
    output = {
        "max_sim_val": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "jaccard": [],
        "rouge1": [],
        "exact_match": [],
    }
    crossed_text = doc["crossed_text"]
    for i in range(len(crossed_text)):
        tmp = vcr_process_results_single(crossed_text[i], results[0], "zh")
        for k in output.keys():
            output[k].append(
                {
                    "question_id": doc["question_id"],
                    "score": tmp[k],
                    "pred_ngram": tmp["max_sim_string"],
                    "gt_ngram": crossed_text[i],
                    "caption": doc["caption"],
                }
            )
    return output


def bootstrap_std(data, n_bootstrap=1000, ci=0.95):
    """
    Args:
        data: a list of values
        n_bootstrap: number of bootstrap samples
        ci: confidence interval
    Returns:
        a tuple of mean, lower bound, upper bound
    """
    n = len(data)
    means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, n, replace=True)
        means.append(np.mean(sample))
    means = np.array(means)
    lower_bound = np.percentile(means, (1 - ci) / 2 * 100)
    upper_bound = np.percentile(means, (1 + ci) / 2 * 100)
    std = np.std(means)
    return std, lower_bound, upper_bound


def vcr_aggregate_results(results, args, metric="exact_match"):
    """
    Args:
        results: List[List[Dict]], list of results returned by process_results
    Returns:
        A float value representing the final score of jaccard index or exact match
    """
    scores = []
    output_dict_detail_result = {}
    for i in range(len(results)):
        for blank_id in range(len(results[i])):
            scores.append(results[i][blank_id]["score"])
        output_dict_detail_result[str(i)] = results[i]
    mean_score = np.mean(scores)
    std, lb, ub = bootstrap_std(scores, n_bootstrap=1000, ci=0.95)
    output_dict = {
        "mean_score": mean_score,
        "std_score": std,
        "percentile_2.5": lb,
        "percentie_97.5": ub,
        "detailed_results": output_dict_detail_result,
    }
    now_date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    path = generate_submission_file(f"vcr_submission_{metric}_{now_date_time}.json", args)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, indent=4, ensure_ascii=False)
    # print(f"Submission file saved to {path}")
    eval_logger.info(f"Submission file saved to {path}")
    return mean_score


def vcr_aggregate_exact_match(results, args):
    return vcr_aggregate_results(results, args, metric="exact_match")


def vcr_aggregate_jaccard(results, args):
    return vcr_aggregate_results(results, args, metric="jaccard")
