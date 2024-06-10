import os
from difflib import SequenceMatcher as SM
import evaluate
import logging
import spacy
from spacy.cli import download
from nltk.util import ngrams
from functools import partial

# Download the English and Chinese models
download("en_core_web_sm")
download("zh_core_web_sm")

eval_logger = logging.getLogger("lmms-eval")

dir_name = os.path.dirname(os.path.abspath(__file__))

rouge = evaluate.load("rouge")
nlp_en = spacy.load("en_core_web_sm")
nlp_zh = spacy.load("zh_core_web_sm")
nlp = {"en": nlp_en, "zh": nlp_zh}

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
    return [doc["stacked_image"].convert("RGB"), doc["only_it_image"].convert("RGB")]


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
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case mme score), value: metric value
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
    ngrams_hasjoint = [
        ngram for ngram in ngrams_ if not set(ngram).isdisjoint(tokens_crossed_text_set)
    ]

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
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case mme score), value: metric value
    """
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    output = {}
    for i in range(len(doc["crossed_text"])):
        res_stacked_image_results = vcr_process_results_single(
            doc["crossed_text"][i], results[0], "en"
        )
        res_only_image_results = vcr_process_results_single(
            doc["crossed_text"][i], results[1], "en"
        )
        output.update(
            {
                f"res_stacked_image__{k}___{i}": v
                for k, v in res_stacked_image_results.items()
            }
        )
        output.update(
            {
                f"res_only_it_image__{k}___{i}": v
                for k, v in res_only_image_results.items()
            }
        )
    return output


def vcr_zh_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case mme score), value: metric value
    """
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    output = {}
    for i in range(len(doc["crossed_text"])):
        res_stacked_image_results = vcr_process_results_single(
            doc["crossed_text"][i], results[0], "zh"
        )
        res_only_image_results = vcr_process_results_single(
            doc["crossed_text"][i], results[1], "zh"
        )
        output.update(
            {
                f"res_stacked_image__{k}___{i}": v
                for k, v in res_stacked_image_results.items()
            }
        )
        output.update(
            {
                f"res_only_it_image__{k}___{i}": v
                for k, v in res_only_image_results.items()
            }
        )
    return output


def vcr_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A dictionary of dictionary of float, where the outer dictionary has keys "res_stacked_image" and "res_only_it_image"
    """
    output = {
        "res_stacked_image__precision": 0,
        "res_stacked_image__recall": 0,
        "res_stacked_image__f1": 0,
        "res_stacked_image__jaccard": 0,
        "res_stacked_image__rouge1": 0,
        "res_stacked_image__exact_match": 0,
        "res_only_it_image__precision": 0,
        "res_only_it_image__recall": 0,
        "res_only_it_image__f1": 0,
        "res_only_it_image__jaccard": 0,
        "res_only_it_image__rouge1": 0,
        "res_only_it_image__exact_match": 0,
    }

    for output_key in output.keys():
        count = 0
        query_domain, query_metric_name = output_key.split("__")
        for inner_dict in results:
            for inner_key, inner_value in inner_dict.items():
                key_domain, key_metric_name, _ = inner_key.split("__")
                if query_domain == key_domain and query_metric_name == key_metric_name:
                    output[output_key] += inner_value
                    count += 1
        output[output_key] /= count
    return output
