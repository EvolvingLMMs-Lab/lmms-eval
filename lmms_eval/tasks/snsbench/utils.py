"""
SNS-Bench-VL: Benchmarking Multimodal Large Language Models in Social Networking Services
- HuggingFace dataset: https://huggingface.co/datasets/morpheushoc/SNS-Bench-VL
- Converted from: https://github.com/HC-Guo/SNS-Bench-VL

@article{guo2025sns,
  title={SNS-Bench-VL: Benchmarking Multimodal Large Language Models in Social Networking Services},
  author={Guo, Hongcheng and Xie, Zheyong and Cao, Shaosheng and Wang, Boyang and Liu, Weiting and Le, Anjie and Li, Lei and Li, Zhoujun},
  journal={arXiv preprint arXiv:2505.23065},
  year={2025}
}
"""

import pandas as pd
from PIL import Image

from lmms_eval.tasks.snsbench import metrics

metrics_map = {
    # MRC
    "note_mrc": "calculate_semantic_similarity",
    # Hash-tag
    "note_hashtag_single": "calculate_answer_accuracy",
    "note_hashtag_multi": "calculate_multiple_choice_f1",
    # Note-Taxonomy
    "note_taxonomy_one_level": "calculate_answer_accuracy",
    "note_taxonomy_three_levels": "calculate_three_level_accuracy",
    # Note-gender
    "note_gender": "calculate_answer_accuracy",
    # Query-corr
    "note_querycorr_two_levels": "calculate_answer_accuracy",
    "note_querycorr_five_levels": "calculate_answer_accuracy",
    # Note-comment
    "note_comment_primary": "calculate_answer_accuracy",
    "note_comment_sub_level": "calculate_answer_accuracy",
    # Note-OCR
    "note_ocr": "calculate_ocr_metrics",
    # Note-Query_Gen
    "note_query_gen": "calculate_answer_accuracy",
}


def doc_to_visual(doc):

    # list of images
    images = doc["images"]
    return images


def doc_to_text(doc, lmms_eval_specific_kwargs=None):

    if "format" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["format"] == "qwen3_vl":
        return doc_to_text_qwen3vl(doc, lmms_eval_specific_kwargs)

    # already contains everything: intro, question and options
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    prompt = f"{pre_prompt}{question}{post_prompt}"
    return prompt


def doc_to_text_qwen3vl(doc, lmms_eval_specific_kwargs=None):
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    question = doc["question"]
    prompt = f"{pre_prompt}{question}{post_prompt}"
    return prompt


def process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name, value: metric value
    """
    pred = results[0]
    gt = doc["answer"]
    category = doc["category"]

    metric_func = getattr(metrics, metrics_map[category])
    score = metric_func(pred, gt)
    return {category: {"question": doc["question"], "prediction": pred, "ground_truth": gt, "score": score}, "average": {"question": doc["question"], "prediction": pred, "ground_truth": gt, "score": score, "category": category}}


def sns_aggregate_results(results):
    df = pd.DataFrame(results)
    avg_score = df["score"].mean()
    return avg_score


if __name__ == "__main__":

    from datasets import load_dataset

    ds = load_dataset("morpheushoc/SNS-Bench-VL")["test"]

    doc = ds[10]
    text = doc_to_text(doc, {})
    lmms_eval_specific_kwargs = {"format": "qwen3_vl", "pre_prompt": "Question: ", "post_prompt": ""}
    text1 = doc_to_text(doc, lmms_eval_specific_kwargs)
    df = ds.to_pandas()
