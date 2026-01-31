"""
Utility functions for Omni Robust Bench evaluation.

This benchmark tests the robustness of omni-modal (vision + audio) video understanding models.
It contains 4 variants:
- standard_vision: Vision questions with correct visual premise
- misleading_vision: Vision questions with WRONG visual premise (tests robustness)
- standard_audio: Audio questions with correct audio premise
- misleading_audio: Audio questions with WRONG audio premise (tests robustness)

For misleading questions, the model should override the incorrect premise in the question
and answer based on what it actually perceives in the video/audio.

Videos are stored at: ngqtrung/video-caption-dataset/videos/{video_id}.mp4
QA data is stored at: ngqtrung/omni_robust_bench

The model receives the full video (which contains both visual and audio tracks).
For omni-modal models like Qwen2.5-Omni, both modalities are automatically processed.
"""

import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from loguru import logger as eval_logger


# Local video cache directory (optional - for faster loading)
VIDEO_CACHE_DIR = os.getenv("OMNI_ROBUST_VIDEO_CACHE", None)


def omni_robust_doc_to_visual(doc):
    """
    Return the video path for this document.

    The video field in the dataset contains the HuggingFace URL to the video file.
    Videos contain both visual and audio tracks - omni-modal models will process both.

    If OMNI_ROBUST_VIDEO_CACHE environment variable is set, will look for local cached videos first.
    """
    video_id = doc["video_id"]

    # Check for local cache first (for faster loading)
    if VIDEO_CACHE_DIR:
        local_path = os.path.join(VIDEO_CACHE_DIR, f"{video_id}.mp4")
        if os.path.exists(local_path):
            return [local_path]

    # Use the video URL from the dataset
    # Format: https://huggingface.co/datasets/ngqtrung/video-caption-dataset/resolve/main/videos/{video_id}.mp4
    video_url = doc.get("video")
    if video_url:
        return [video_url]

    # Fallback: construct URL from video_id
    return [f"https://huggingface.co/datasets/ngqtrung/video-caption-dataset/resolve/main/videos/{video_id}.mp4"]


def omni_robust_doc_to_text_vision(doc, lmms_eval_specific_kwargs=None):
    """
    Format the question text with choices for vision questions.

    Uses the new dataset format with option_a, option_b, option_c, option_d fields.
    """
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get(
        "post_prompt",
        "Answer with the option's letter from the given choices directly."
    )

    question = doc["question"]

    # Get choices from individual option fields
    options_text = f"A. {doc.get('option_a', '')}\nB. {doc.get('option_b', '')}\nC. {doc.get('option_c', '')}\nD. {doc.get('option_d', '')}"

    prompt_text = (
        f"{pre_prompt}"
        f"Question:\n{question}\n\n"
        f"Options:\n{options_text}\n\n"
        f"{post_prompt}"
    )

    return prompt_text


def omni_robust_doc_to_text_audio(doc, lmms_eval_specific_kwargs=None):
    """
    Format the question text with choices for audio questions.

    Uses the new dataset format with option_a, option_b, option_c, option_d fields.
    """
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get(
        "post_prompt",
        "Answer with the option's letter from the given choices directly."
    )

    question = doc["question"]

    # Get choices from individual option fields
    options_text = f"A. {doc.get('option_a', '')}\nB. {doc.get('option_b', '')}\nC. {doc.get('option_c', '')}\nD. {doc.get('option_d', '')}"

    prompt_text = (
        f"{pre_prompt}"
        f"Question:\n{question}\n\n"
        f"Options:\n{options_text}\n\n"
        f"{post_prompt}"
    )

    return prompt_text


def omni_robust_doc_to_choice_vision(doc):
    """Return list of choices for vision questions."""
    return [doc.get('option_a', ''), doc.get('option_b', ''), doc.get('option_c', ''), doc.get('option_d', '')]


def omni_robust_doc_to_choice_audio(doc):
    """Return list of choices for audio questions."""
    return [doc.get('option_a', ''), doc.get('option_b', ''), doc.get('option_c', ''), doc.get('option_d', '')]


def omni_robust_doc_to_target(doc):
    """
    Return the target answer index for standard questions.
    For standard questions, correct_answer is always "A" (answer index 0).
    """
    correct_answer = doc.get("correct_answer")
    if correct_answer is None:
        return 0  # Default to A if not specified

    letter_to_index = {"A": 0, "B": 1, "C": 2, "D": 3}
    return letter_to_index.get(correct_answer, 0)


def omni_robust_doc_to_target_misleading(doc):
    """
    Return the target answer index for misleading questions.

    For misleading questions, the correct_answer field may be null because
    the question contains a wrong premise. However, the "actual" correct answer
    (what the model should select based on real perception) is still "A".

    The benchmark tests whether the model can override the misleading premise.
    """
    # For misleading questions, the model should still select A
    # because A is the correct answer based on actual video/audio content
    return 0


def parse_multi_choice_response(response, all_choices):
    """
    Extract the choice letter from model response.

    Handles various formats:
    - "A", "B", "C", "D"
    - "(A)", "(B)", etc.
    - "A.", "B.", etc.
    - "The answer is A", "Answer: B", etc.
    """
    response = " " + response.strip() + " "
    candidates = []
    ans_with_brack = False

    # Try format: (A), (B), etc.
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    # Try format: A , B , etc. (with spaces)
    if not candidates:
        for choice in all_choices:
            if f" {choice} " in response:
                candidates.append(choice)

    # Try format: A., B., etc.
    if not candidates:
        for choice in all_choices:
            if f"{choice}." in response:
                candidates.append(choice)

    if not candidates:
        return None

    # If multiple candidates, take the last occurrence (usually the final answer)
    if len(candidates) > 1:
        if ans_with_brack:
            start_indexes = [response.rfind(f"({c})") for c in candidates]
        else:
            start_indexes = [response.rfind(f" {c} ") for c in candidates]
        return candidates[np.argmax(start_indexes)]

    return candidates[0]


def omni_robust_process_results(doc, results):
    """
    Process results for standard questions.

    Returns accuracy dict with overall score and category breakdown.
    """
    response_text = results[0].strip()
    all_choices = ["A", "B", "C", "D"]

    pred = parse_multi_choice_response(response_text, all_choices)

    # Get ground truth
    correct_answer = doc.get("correct_answer", "A")

    # Calculate correctness
    is_correct = 1 if pred == correct_answer else 0

    # Get category for breakdown
    category = doc.get("category", "unknown")
    modality = doc.get("modality", "unknown")

    return {
        "accuracy": {
            "overall": is_correct,
            "category": category,
            "modality": modality,
            "pred": pred,
            "gold": correct_answer,
            "video_id": doc.get("video_id", ""),
        }
    }


def omni_robust_process_results_misleading(doc, results):
    """
    Process results for misleading questions.

    For misleading questions, we evaluate whether the model:
    1. Overrides the wrong premise and selects A (the actual correct answer) - ROBUST
    2. Gets confused by the wrong premise - NOT ROBUST

    We track both:
    - Accuracy: Did the model select A (the correct answer based on actual content)?
    - Robustness: Did the model resist being misled?
    """
    response_text = results[0].strip()
    all_choices = ["A", "B", "C", "D"]

    pred = parse_multi_choice_response(response_text, all_choices)

    # For misleading questions, the actual correct answer is A
    # (based on the real video/audio content, not the misleading premise)
    correct_answer = "A"

    # Calculate correctness (robustness)
    is_robust = 1 if pred == correct_answer else 0

    # Get category for breakdown
    category = doc.get("category", "unknown")
    modality = doc.get("modality", "unknown")
    misleading_category = doc.get("misleading_category", category)

    return {
        "accuracy": {
            "overall": is_robust,
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
    """
    total_correct = 0
    total_examples = 0

    category_correct = defaultdict(int)
    category_total = defaultdict(int)

    modality_correct = defaultdict(int)
    modality_total = defaultdict(int)

    misleading_category_correct = defaultdict(int)
    misleading_category_total = defaultdict(int)

    for result in results:
        overall_score = result.get("overall", 0)
        total_correct += overall_score
        total_examples += 1

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
    eval_logger.info(f"OMNI ROBUST BENCH RESULTS")
    eval_logger.info("=" * 60)
    eval_logger.info(f"Overall Accuracy: {overall_accuracy:.2f}% ({total_correct}/{total_examples})")

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
