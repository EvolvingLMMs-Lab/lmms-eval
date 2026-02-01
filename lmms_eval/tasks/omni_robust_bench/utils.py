"""
Utility functions for Omni Robust Bench evaluation.

This benchmark tests the robustness of omni-modal (vision + audio) video understanding models.
It contains 4 variants:
- standard_vision: Vision questions with correct visual premise
- misleading_vision: Vision questions with WRONG visual premise (tests robustness)
- standard_audio: Audio questions with correct audio premise
- misleading_audio: Audio questions with WRONG audio premise (tests robustness)

For ALL questions, there are 6 choices (A-F):
- A-D: Content-based choices
- E: "The visual detail in the question is incorrect"
- F: "The audio detail in the question is incorrect"

For standard questions, correct answer is A (content-based).
For misleading_vision questions, correct answer is E (visual premise is wrong).
For misleading_audio questions, correct answer is F (audio premise is wrong).

ALL variants receive video input (both audio and vision tracks).
"""

import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from loguru import logger as eval_logger


# Local video cache directory (optional - for faster loading)
VIDEO_CACHE_DIR = os.getenv("OMNI_ROBUST_VIDEO_CACHE", None)

# All 6 choice letters
ALL_CHOICES = ["A", "B", "C", "D", "E", "F"]

# Letter to index mapping
LETTER_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}


def omni_robust_doc_to_visual(doc):
    """
    Return the video path for this document.

    The video field in the dataset contains the path/URL to the video file.
    Videos contain both visual and audio tracks - omni-modal models will process both.

    If OMNI_ROBUST_VIDEO_CACHE environment variable is set, will look for local cached videos first.
    """
    video_id = doc["video_id"]

    # Check for local cache first (for faster loading)
    if VIDEO_CACHE_DIR:
        local_path = os.path.join(VIDEO_CACHE_DIR, f"{video_id}.mp4")
        if os.path.exists(local_path):
            return [local_path]

    # Use the video from the dataset
    video = doc.get("video")
    if video:
        # Handle both path strings and Video objects
        if isinstance(video, dict) and "path" in video:
            return [video["path"]]
        return [video]

    # Fallback: construct URL from video_id
    return [f"https://huggingface.co/datasets/ngqtrung/video-caption-dataset/resolve/main/videos/{video_id}.mp4"]


def omni_robust_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """
    Format the question text with all 6 choices (A-F).
    Used for ALL variants - both vision and audio questions.
    """
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get(
        "post_prompt",
        "Answer with the option's letter from the given choices directly."
    )

    question = doc["question"]

    # Get all 6 choices
    options_text = (
        f"A. {doc.get('option_a', '')}\n"
        f"B. {doc.get('option_b', '')}\n"
        f"C. {doc.get('option_c', '')}\n"
        f"D. {doc.get('option_d', '')}\n"
        f"E. {doc.get('option_e', 'The visual detail in the question is incorrect')}\n"
        f"F. {doc.get('option_f', 'The audio detail in the question is incorrect')}"
    )

    prompt_text = (
        f"{pre_prompt}"
        f"Question:\n{question}\n\n"
        f"Options:\n{options_text}\n\n"
        f"{post_prompt}"
    )

    return prompt_text


# Alias functions for backward compatibility with YAML configs
def omni_robust_doc_to_text_vision(doc, lmms_eval_specific_kwargs=None):
    """Format question for vision variants (same as generic)."""
    return omni_robust_doc_to_text(doc, lmms_eval_specific_kwargs)


def omni_robust_doc_to_text_audio(doc, lmms_eval_specific_kwargs=None):
    """Format question for audio variants (same as generic)."""
    return omni_robust_doc_to_text(doc, lmms_eval_specific_kwargs)


def omni_robust_doc_to_choice(doc):
    """Return list of all 6 choices."""
    return [
        doc.get('option_a', ''),
        doc.get('option_b', ''),
        doc.get('option_c', ''),
        doc.get('option_d', ''),
        doc.get('option_e', 'The visual detail in the question is incorrect'),
        doc.get('option_f', 'The audio detail in the question is incorrect'),
    ]


# Alias functions for backward compatibility
def omni_robust_doc_to_choice_vision(doc):
    """Return list of choices for vision questions."""
    return omni_robust_doc_to_choice(doc)


def omni_robust_doc_to_choice_audio(doc):
    """Return list of choices for audio questions."""
    return omni_robust_doc_to_choice(doc)


def omni_robust_doc_to_target(doc):
    """
    Return the target answer index.

    For standard questions: A (index 0)
    For misleading_vision: E (index 4)
    For misleading_audio: F (index 5)
    """
    correct_answer = doc.get("correct_answer", "A")
    return LETTER_TO_INDEX.get(correct_answer, 0)


def omni_robust_doc_to_target_misleading(doc):
    """
    Return the target answer index for misleading questions.

    For misleading_vision: E (index 4) - visual premise is wrong
    For misleading_audio: F (index 5) - audio premise is wrong
    """
    correct_answer = doc.get("correct_answer")
    if correct_answer:
        return LETTER_TO_INDEX.get(correct_answer, 4)

    # Fallback based on modality
    modality = doc.get("modality", "vision")
    if modality == "vision":
        return 4  # E
    else:
        return 5  # F


def get_multi_choice_info(options):
    """
    Given the list of options for multiple choice question
    Return the index2ans and all_choices.

    Adapted from MMMU implementation:
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/data_utils.py#L54
    """
    start_chr = "A"
    all_choices = []
    index2ans = {}
    for i, option in enumerate(options):
        index2ans[chr(ord(start_chr) + i)] = option
        all_choices.append(chr(ord(start_chr) + i))
    return index2ans, all_choices


def parse_multi_choice_response(response, all_choices=None, index2ans=None):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D, E, F.

    Adapted from MMMU implementation:
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L10
    """
    if all_choices is None:
        all_choices = ALL_CHOICES

    # Strip punctuation from response
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []

    # Try format: (A), (B), etc.
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    # Try format: A , B , etc. (with space)
    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice} " in response:
                candidates.append(choice)

    # Try format: A., B., etc.
    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice}." in response:
                candidates.append(choice)

    # If no candidates found and response is long, try matching answer content
    if len(candidates) == 0 and len(response.split()) > 5 and index2ans:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content answer

    # If still no candidates, randomly choose one
    if len(candidates) == 0:
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        # Multiple candidates - take the last occurrence (usually the final answer)
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # Get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else:
        # Only one candidate, use it
        pred_index = candidates[0]

    return pred_index


def omni_robust_process_results(doc, results):
    """
    Process results for all question types.

    Returns accuracy dict with overall score and category breakdown.
    """
    response_text = results[0].strip()

    # Get options and create index2ans mapping for content matching
    options = [
        doc.get('option_a', ''),
        doc.get('option_b', ''),
        doc.get('option_c', ''),
        doc.get('option_d', ''),
        doc.get('option_e', 'The visual detail in the question is incorrect'),
        doc.get('option_f', 'The audio detail in the question is incorrect'),
    ]
    index2ans, all_choices = get_multi_choice_info(options)

    pred = parse_multi_choice_response(response_text, all_choices, index2ans)

    # Get ground truth
    correct_answer = doc.get("correct_answer", "A")

    # Calculate correctness
    is_correct = 1 if pred == correct_answer else 0

    # Get metadata for breakdown
    category = doc.get("category", "unknown")
    modality = doc.get("modality", "unknown")
    is_misleading = doc.get("is_misleading", False)
    misleading_category = doc.get("misleading_category", category)

    return {
        "accuracy": {
            "overall": is_correct,
            "category": category,
            "modality": modality,
            "pred": pred,
            "gold": correct_answer,
            "video_id": doc.get("video_id", ""),
            "is_misleading": is_misleading,
            "misleading_category": misleading_category if is_misleading else "",
        }
    }


def omni_robust_process_results_misleading(doc, results):
    """
    Process results for misleading questions.

    For misleading questions:
    - Vision: correct answer is E (visual detail is incorrect)
    - Audio: correct answer is F (audio detail is incorrect)
    """
    response_text = results[0].strip()

    # Get options and create index2ans mapping for content matching
    options = [
        doc.get('option_a', ''),
        doc.get('option_b', ''),
        doc.get('option_c', ''),
        doc.get('option_d', ''),
        doc.get('option_e', 'The visual detail in the question is incorrect'),
        doc.get('option_f', 'The audio detail in the question is incorrect'),
    ]
    index2ans, all_choices = get_multi_choice_info(options)

    pred = parse_multi_choice_response(response_text, all_choices, index2ans)

    # Get ground truth (E for vision, F for audio)
    correct_answer = doc.get("correct_answer")
    if not correct_answer:
        modality = doc.get("modality", "vision")
        correct_answer = "E" if modality == "vision" else "F"

    # Calculate correctness
    is_correct = 1 if pred == correct_answer else 0

    # Get metadata
    category = doc.get("category", "unknown")
    modality = doc.get("modality", "unknown")
    misleading_category = doc.get("misleading_category", category)

    return {
        "accuracy": {
            "overall": is_correct,
            "category": category,
            "modality": modality,
            "misleading_category": misleading_category,
            "pred": pred,
            "gold": correct_answer,
            "video_id": doc.get("video_id", ""),
            "is_misleading": True,
        }
    }


def omni_robust_aggregate_results(results):
    """
    Aggregate accuracy across all examples.

    Computes:
    - Overall accuracy
    - Accuracy by category
    - Accuracy by modality
    - For misleading: robustness metrics
    """
    total_correct = 0
    total_examples = 0

    category_correct = defaultdict(int)
    category_total = defaultdict(int)

    modality_correct = defaultdict(int)
    modality_total = defaultdict(int)

    misleading_category_correct = defaultdict(int)
    misleading_category_total = defaultdict(int)

    # Track prediction distribution
    pred_distribution = defaultdict(int)

    for result in results:
        overall_score = result.get("overall", 0)
        total_correct += overall_score
        total_examples += 1

        # Track predictions
        pred = result.get("pred", "None")
        pred_distribution[pred] += 1

        # Track by category
        category = result.get("category", "unknown")
        category_correct[category] += overall_score
        category_total[category] += 1

        # Track by modality
        modality = result.get("modality", "unknown")
        modality_correct[modality] += overall_score
        modality_total[modality] += 1

        # Track by misleading category (if present)
        if result.get("is_misleading"):
            mis_cat = result.get("misleading_category", "unknown")
            misleading_category_correct[mis_cat] += overall_score
            misleading_category_total[mis_cat] += 1

    # Compute overall accuracy
    overall_accuracy = (total_correct / total_examples) * 100 if total_examples > 0 else 0.0

    # Compute category accuracy
    category_accuracy = {
        cat: (category_correct[cat] / category_total[cat]) * 100
        if category_total[cat] > 0 else 0.0
        for cat in category_correct
    }

    # Compute modality accuracy
    modality_accuracy = {
        mod: (modality_correct[mod] / modality_total[mod]) * 100
        if modality_total[mod] > 0 else 0.0
        for mod in modality_correct
    }

    # Log results
    eval_logger.info("=" * 60)
    eval_logger.info("OMNI ROBUST BENCH RESULTS (6 choices: A-F)")
    eval_logger.info("=" * 60)
    eval_logger.info(f"Overall Accuracy: {overall_accuracy:.2f}% ({total_correct}/{total_examples})")

    # Log prediction distribution
    eval_logger.info("\nPrediction Distribution:")
    for choice in ALL_CHOICES:
        count = pred_distribution.get(choice, 0)
        pct = (count / total_examples * 100) if total_examples > 0 else 0
        eval_logger.info(f"  {choice}: {count} ({pct:.1f}%)")

    if modality_accuracy:
        eval_logger.info("\nAccuracy by Modality:")
        for modality, acc in sorted(modality_accuracy.items()):
            count = modality_total[modality]
            correct = modality_correct[modality]
            eval_logger.info(f"  {modality}: {acc:.2f}% ({correct}/{count})")

    if category_accuracy:
        eval_logger.info("\nAccuracy by Category (top 10):")
        sorted_cats = sorted(category_accuracy.items(), key=lambda x: -x[1])[:10]
        for cat, acc in sorted_cats:
            count = category_total[cat]
            correct = category_correct[cat]
            eval_logger.info(f"  {cat}: {acc:.2f}% ({correct}/{count})")

    if misleading_category_total:
        eval_logger.info("\nRobustness by Misleading Category:")
        for cat in sorted(misleading_category_total.keys()):
            count = misleading_category_total[cat]
            correct = misleading_category_correct[cat]
            acc = (correct / count) * 100 if count > 0 else 0
            eval_logger.info(f"  {cat}: {acc:.2f}% ({correct}/{count})")

    eval_logger.info("=" * 60)

    return round(overall_accuracy, 5)
