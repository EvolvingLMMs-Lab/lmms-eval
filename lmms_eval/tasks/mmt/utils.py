import json
import random
from collections import defaultdict

import numpy as np
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

# All abbreviations for the categories
"""
MMT_abbrs = {
    'visual_recognition': 'VR',
    'localization': 'Loc',
    'ocr': 'OCR',
    'counting': 'Count',
    'hallucination': 'HLN',
    'image_retrieval': 'IR',
    'threed': '3D',
    'visual_captioning': 'VC',
    'visual_grounding': 'VG',
    'doc_understanding': 'DU',
    'action_recognition': 'AR',
    'pixel_level_perception': 'PLP',
    'image-to-image_translation': 'I2IT',
    'relation_reasoning': 'RR',
    'intelligence_quotient_test': 'IQT',
    'emotion': 'Emo',
    'visual_illusion': 'VI',
    'meme_understanding': 'MemU',
    'visual_prompt_understanding': 'VPU',
    'anomaly_detection': 'AND',
    'keypoint_detection': 'KD',
    'visual_commonsense_reasoning': 'VCR',
    'image_evaluation_judgement': 'IEJ',
    'multiple_image_analysis': 'MIA',
    'cross_image_matching': 'CIM',
    'temporal_understanding': 'TU',
    'visual_code': 'VP',
    'medical_understanding': 'MedU',
    'autonomous_driving': 'AUD',
    'discipline_knowledge_reasoning': 'DKR',
    'embodied_ai': 'EA',
    'gui_navigation': 'GN'
}
"""
# ============================
# Visual Processing Functions
# ============================


def mmt_doc_to_visual(doc):
    return [image.convert("RGB") for image in doc["image"]]

# ============================
# Text Processing Functions
# ============================


def mmt_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question_text = "Question: <image>\n" + doc["question"].strip()

    options = []
    for option in ["A", "B", "C", "D", "E", "F", "G", "H", "I"]:
        option_text = doc.get(option)
        if option_text and option_text.strip():
            options.append(f"{option}: {option_text.strip()}")

    options_text = "\n".join(options) if options else ""

    formatted_question = f"{question_text}\n{options_text}"

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") if lmms_eval_specific_kwargs else ""

    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "") if lmms_eval_specific_kwargs else ""
    formatted_question = f"{pre_prompt}{formatted_question}{post_prompt}"

    return formatted_question


def mmt_doc_to_choice(doc):
    choices = []
    for option in ["A", "B", "C", "D", "E", "F", "G", "H", "I"]:
        if doc.get(option) and doc[option].strip():
            choices.append(option)
    return choices


def mmt_doc_to_target(doc):
    return doc["answer"]


# ============================
# Result Processing Functions
# ============================


def mmt_process_results(doc, results):
    response = results[0].strip()
    all_choices = [choice for choice in ["A", "B", "C", "D", "E", "F", "G", "H", "I"] if doc.get(choice)]
    pred = parse_multi_choice_response(response, all_choices)  # AdaptfromMMMU
    gt_ans = doc.get("answer", "").strip()
    score = 1.0 if pred == gt_ans else 0.0
    l2_category = doc.get("l2-category", "unknown")

    accuracy_dict = {"overall": score, l2_category: score}  # Overall accuracy  # Accuracy for the specific sub-category
    submission_dict = {}
    if doc.get("split") == "TEST":
        submission_dict = {doc.get("index", "unknown"): pred}  # Save using index
    return {"accuracy": accuracy_dict, "submission": submission_dict}


# ============================
# Aggregation Functions
# ============================


def mmt_aggregate_results(results):
    total_correct = 0
    total_examples = 0
    category_correct = defaultdict(int)
    category_total = defaultdict(int)
    for result in results:
        # Overall accuracy
        total_correct += result["overall"]
        total_examples += 1
        # Sub-category accuracy
        for category, score in result.items():
            if category != "overall":
                category_correct[category] += score
                category_total[category] += 1
    overall_accuracy = (total_correct / total_examples) * 100 if total_examples > 0 else 0.0
    category_accuracy = {category: ((category_correct[category] / category_total[category]) * 100 if category_total[category] > 0 else 0.0) for category in category_correct}
    final_results = {"overall_accuracy": round(overall_accuracy, 5), "category_accuracy": {category: round(acc, 5) for category, acc in category_accuracy.items()}}
    print(final_results)
    return final_results


def mmt_test_aggregate_results_for_submission(results, args):
    path = generate_submission_file("mmt_test_submission.json", args)
    with open(path, "w") as f:
        json.dump(results, f)
    eval_logger.info(f"Results saved to {path}.")


##################
# Helper functions adapted from MMMU's utils.py.
##################


def parse_multi_choice_response(response, all_choices):
    """
    Parse the prediction from the generated response.
    Return the predicted choice letter e.g., A, B, C, D.
    """
    # Clean response of unwanted characters
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # Add space to avoid partial match

    candidates = []
    # Look for choices with parentheses, e.g., (A)
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)

    # Look for simple choices, e.g., A, B, C
    if len(candidates) == 0:
        for choice in all_choices:
            if f" {choice} " in response:
                candidates.append(choice)

    # Look for choices with periods, e.g., A., B., C.
    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice}." in response:
                candidates.append(choice)

    # If no candidates, randomly choose one
    if len(candidates) == 0:
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        # If more than one candidate, choose the last one found
        start_indexes = [response.rfind(f" {can} ") for can in candidates]
        pred_index = candidates[np.argmax(start_indexes)]
    else:
        # If only one candidate, use it
        pred_index = candidates[0]

    return pred_index
