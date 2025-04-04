import logging
import random
from collections import defaultdict

import numpy as np
from loguru import logger as eval_logger


def omni_bench_original_doc_to_visual(doc):
    return [doc["audio"], doc["image"]]


def omni_bench_original_doc_to_text(doc, lmms_eval_specific_kwargs):
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    question = doc["question"]
    options = doc["options"]
    letters = ["A", "B", "C", "D"]
    enumerated_opts = [f"{letters[i]}. {opt}" for i, opt in enumerate(options)]
    prompt_text = f"{pre_prompt}" f"Question:\n{question}\n\n" f"Options:\n" + "\n".join(enumerated_opts) + f"\n\n{post_prompt}"
    return prompt_text


def omni_bench_image_caption_doc_to_visual(doc):
    return [doc["audio"]]


def omni_bench_image_caption_doc_to_text(doc, lmms_eval_specific_kwargs):
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    image_caption = doc["image content"]
    question = doc["question"]
    options = doc["options"]
    letters = ["A", "B", "C", "D"]
    enumerated_opts = [f"{letters[i]}. {opt}" for i, opt in enumerate(options)]
    prompt_text = f"{pre_prompt}" f"Visual Context for the audio:\n{image_caption}\n\n" f"Question:\n{question}\n\n" f"Options:\n" + "\n".join(enumerated_opts) + "\n\n" f"{post_prompt}"
    return prompt_text


def omni_bench_audio_transcript_doc_to_visual(doc):
    return [doc["image"]]


def omni_bench_audio_transcript_doc_to_text(doc, lmms_eval_specific_kwargs):
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    audio_transcript = doc.get("audio content", "")
    question = doc["question"]
    options = doc["options"]
    letters = ["A", "B", "C", "D"]
    enumerated_opts = [f"{letters[i]}. {opt}" for i, opt in enumerate(options)]
    prompt_text = f"{pre_prompt}" f"Audio context for the images:\n{audio_transcript}\n\n" f"Question:\n{question}\n\n" f"Options:\n" + "\n".join(enumerated_opts) + "\n\n" f"{post_prompt}"
    return prompt_text


def omni_bench_doc_to_choice(doc):
    return doc["options"]


def omni_bench_doc_to_target(doc):
    correct_answer_str = doc["answer"]
    return doc["options"].index(correct_answer_str)


def omni_bench_process_results(doc, results):
    response_text = results[0].strip()
    all_choices = ["A", "B", "C", "D"]
    pred = parse_multi_choice_response(response_text, all_choices)
    letter_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    gold_index = omni_bench_doc_to_target(doc)
    gold_letter = letter_map.get(gold_index, "X")
    overall_correct = 1 if pred == gold_letter else 0
    category = doc.get("task type", "unknown")
    accuracy_dict = {"overall": overall_correct, category: overall_correct}
    return {"accuracy": accuracy_dict}


def omni_bench_aggregate_results(results):
    total_correct = 0
    total_examples = 0
    category_correct = defaultdict(int)
    category_total = defaultdict(int)
    for i, result in enumerate(results, 1):
        overall_score = result["overall"]
        total_correct += overall_score
        total_examples += 1
        for category, score in result.items():
            if category != "overall":
                category_correct[category] += score
                category_total[category] += 1
    overall_accuracy = (total_correct / total_examples) * 100 if total_examples > 0 else 0.0
    category_accuracy = {category: (category_correct[category] / category_total[category]) * 100 if category_total[category] > 0 else 0.0 for category in category_correct}
    eval_logger.info("=" * 50)
    eval_logger.info(f"Overall Accuracy: {overall_accuracy:.2f}%")

    if category_accuracy:
        eval_logger.info("Category-wise Accuracy:")
        for category, acc in category_accuracy.items():
            eval_logger.info(f"  {category}: {acc:.2f}%")
    eval_logger.info("=" * 50)
    return round(overall_accuracy, 5)


def parse_multi_choice_response(response, all_choices):
    response = " " + response.strip() + " "
    index_ans = True
    ans_with_brack = False
    candidates = []

    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if not candidates:
        for choice in all_choices:
            if f"{choice} " in response:
                candidates.append(choice)

    if not candidates:
        for choice in all_choices:
            if f"{choice}." in response:
                candidates.append(choice)

    if not candidates:
        return None

    if len(candidates) > 1:
        start_indexes = []
        if ans_with_brack:
            for can in candidates:
                start_indexes.append(response.rfind(f"({can})"))
        else:
            for can in candidates:
                start_indexes.append(response.rfind(f" {can} "))
        return candidates[np.argmax(start_indexes)]
    else:
        return candidates[0]
