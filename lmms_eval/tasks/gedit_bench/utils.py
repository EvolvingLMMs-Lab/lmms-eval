"""
GEdit-Bench Utils
Image editing evaluation task using VIEScore
"""

import json
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from loguru import logger as eval_logger
from PIL import Image

# Try to import VIEScore
from lmms_eval.tasks.gedit_bench.viescore import VIEScore

# Task groups for GEdit-Bench
GEDIT_BENCH_GROUPS = [
    "background_change",
    "color_alter",
    "material_alter",
    "motion_change",
    "ps_human",
    "style_change",
    "subject-add",
    "subject-remove",
    "subject-replace",
    "text_change",
    "tone_transfer",
]


def calculate_dimensions(target_area, ratio):
    """Calculate dimensions maintaining aspect ratio"""
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    new_area = width * height
    return int(width), int(height), int(new_area)


def _get_vie_score():
    """
    Get or create VIEScore instance.
    Note: In multi-process environments, each process will have its own instance.
    """
    return VIEScore(task="tie")


def gedit_bench_doc_to_visual(doc):
    """Extract input image from document"""
    # Try different possible field names
    input_image = doc.get("input_image") or doc.get("input_image_raw")
    if input_image is None:
        eval_logger.warning(f"No input image found in document. Available keys: {list(doc.keys())}")
        return []
    # Convert to RGB if it's a PIL Image
    if hasattr(input_image, "convert"):
        return [input_image.convert("RGB")]
    return [input_image]


def gedit_bench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Extract instruction text from document"""
    instruction = doc.get("instruction", "").strip()
    pre_prompt = ""
    post_prompt = ""
    if lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    return f"{pre_prompt}{instruction}{post_prompt}"


def gedit_bench_doc_to_target(doc):
    """Extract target instruction (for reference)"""
    return doc.get("instruction", "")


def _create_all_metric_results(key, task_type, instruction_language, semantics_score, quality_score, overall_score, intersection_exist):
    """
    Create result dict with all metric keys for detailed breakdown.

    Returns metrics for:
    - Overall scores (all samples)
    - English fullset and intersection
    - Chinese fullset and intersection
    """
    base_entry = {
        "key": key,
        "task_type": task_type,
        "instruction_language": instruction_language,
        "intersection_exist": intersection_exist,
    }

    return {
        # Overall scores (used for global aggregation)
        "gedit_bench_semantics_score": {**base_entry, "score": semantics_score},
        "gedit_bench_quality_score": {**base_entry, "score": quality_score},
        "gedit_bench_overall_score": {**base_entry, "score": overall_score},
        # English fullset metrics
        "gedit_bench_en_fullset_semantics": {**base_entry, "score": semantics_score},
        "gedit_bench_en_fullset_quality": {**base_entry, "score": quality_score},
        "gedit_bench_en_fullset_overall": {**base_entry, "score": overall_score},
        # English intersection metrics
        "gedit_bench_en_intersection_semantics": {**base_entry, "score": semantics_score},
        "gedit_bench_en_intersection_quality": {**base_entry, "score": quality_score},
        "gedit_bench_en_intersection_overall": {**base_entry, "score": overall_score},
        # Chinese fullset metrics
        "gedit_bench_cn_fullset_semantics": {**base_entry, "score": semantics_score},
        "gedit_bench_cn_fullset_quality": {**base_entry, "score": quality_score},
        "gedit_bench_cn_fullset_overall": {**base_entry, "score": overall_score},
        # Chinese intersection metrics
        "gedit_bench_cn_intersection_semantics": {**base_entry, "score": semantics_score},
        "gedit_bench_cn_intersection_quality": {**base_entry, "score": quality_score},
        "gedit_bench_cn_intersection_overall": {**base_entry, "score": overall_score},
    }


def gedit_bench_process_results(doc, results, **kwargs):
    """
    Process model predictions:
    1. Parse JSON output to extract text and images from unified model output path
    2. Reorganize images to GEdit-Bench required directory structure
    3. Evaluate using VIEScore

    Model saves images to: {output_dir}/gedit_bench/{key}.png
    GEdit-Bench requires: {output_dir}/{model_name}/fullset/{task_type}/{language}/{key}.png

    Args:
        doc: Document containing input image, instruction, key, task_type, etc.
        results: Model predictions [JSON string with {"text": "...", "images": [...]}]
        **kwargs: Additional arguments (may include full_docs)

    Returns:
        Dict with metrics for all breakdown categories
    """

    pred = results[0]
    try:
        pred = json.loads(pred)
    except json.JSONDecodeError:
        eval_logger.warning(f"Failed to parse prediction JSON: {pred}")
        pred = {"text": "", "images": []}

    model_text = pred.get("text", "")
    model_images = pred.get("images", [])

    # Extract document fields
    key = doc.get("key", "unknown")
    task_type = doc.get("task_type", "unknown")
    instruction = doc.get("instruction", "")
    instruction_language = doc.get("instruction_language", "en")
    intersection_exist = doc.get("Intersection_exist", False)
    input_image_pil = doc.get("input_image")

    # Evaluate using VIEScore
    try:
        # Load edited image
        edited_image_pil = Image.open(model_images[0]).convert("RGB")

        # Resize images to target area (512x512 equivalent)
        source_img_width, source_img_height, _ = calculate_dimensions(512 * 512, input_image_pil.width / input_image_pil.height)
        edited_img_width, edited_img_height, _ = calculate_dimensions(512 * 512, edited_image_pil.width / edited_image_pil.height)

        input_image_pil = input_image_pil.resize((source_img_width, source_img_height))
        edited_image_pil = edited_image_pil.resize((edited_img_width, edited_img_height))

        # Get VIEScore instance
        vie_score = _get_vie_score()

        # Evaluate: VIEScore.evaluate returns [semantics_score, quality_score, overall_score]
        score_list = vie_score.evaluate([input_image_pil, edited_image_pil], instruction)
        semantics_score, quality_score, overall_score = score_list

        return _create_all_metric_results(key, task_type, instruction_language, float(semantics_score), float(quality_score), float(overall_score), intersection_exist)
    except Exception as e:
        eval_logger.error(f"Error evaluating key {key}: {e}")
        return _create_all_metric_results(key, task_type, instruction_language, 0.0, 0.0, 0.0, intersection_exist)


def gedit_bench_aggregate_results(results):
    """
    Aggregate results across all samples and compute final scores.
    Also logs detailed breakdown by task type, language, and intersection status.

    Args:
        results: List of result dicts from process_results, each containing:
            - key: Sample key
            - task_type: Task type
            - instruction_language: Language ("en" or "cn")
            - score: Score value
            - intersection_exist: Whether intersection exists

    Returns:
        Final aggregated score (average across all samples)
    """
    if not results:
        return 0.0

    # Calculate average score
    scores = [r["score"] for r in results if "score" in r]
    if not scores:
        return 0.0

    avg_score = np.mean(scores)

    # Log breakdown by task type and language
    task_type_scores = defaultdict(list)
    language_scores = defaultdict(list)
    intersection_scores = []
    non_intersection_scores = []

    for r in results:
        if "score" in r:
            task_type = r.get("task_type", "unknown")
            language = r.get("instruction_language", "unknown")
            intersection_exist = r.get("intersection_exist", False)

            task_type_scores[task_type].append(r["score"])
            language_scores[language].append(r["score"])

            if intersection_exist:
                intersection_scores.append(r["score"])
            else:
                non_intersection_scores.append(r["score"])

    # Log statistics
    eval_logger.info(f"Overall average score: {avg_score:.4f}")
    eval_logger.info(f"Number of samples: {len(scores)}")

    if task_type_scores:
        eval_logger.info("Scores by task type:")
        for task_type, task_scores in sorted(task_type_scores.items()):
            task_avg = np.mean(task_scores)
            eval_logger.info(f"  {task_type}: {task_avg:.4f} (n={len(task_scores)})")

    if language_scores:
        eval_logger.info("Scores by language:")
        for language, lang_scores in sorted(language_scores.items()):
            lang_avg = np.mean(lang_scores)
            eval_logger.info(f"  {language}: {lang_avg:.4f} (n={len(lang_scores)})")

    if intersection_scores:
        intersection_avg = np.mean(intersection_scores)
        eval_logger.info(f"Intersection samples average: {intersection_avg:.4f} (n={len(intersection_scores)})")

    if non_intersection_scores:
        non_intersection_avg = np.mean(non_intersection_scores)
        eval_logger.info(f"Non-intersection samples average: {non_intersection_avg:.4f} (n={len(non_intersection_scores)})")

    return float(avg_score)


# ============================================
# Helper Function for Filtered Aggregation
# ============================================


def _aggregate_by_filter(results, language: str = None, intersection_only: bool = None):
    """
    Helper function to aggregate scores with filters.

    Args:
        results: List of result dicts
        language: Filter by language ("en" or "cn"), None for all
        intersection_only: True for intersection subset only, None for all (fullset)

    Returns:
        Average score for filtered samples
    """
    if not results:
        return 0.0

    filtered_scores = []
    for r in results:
        if "score" not in r:
            continue

        # Apply language filter
        if language is not None:
            if r.get("instruction_language", "unknown") != language:
                continue

        # Apply intersection filter (only filter if intersection_only is True)
        # intersection_only=None means fullset (all samples of that language)
        # intersection_only=True means only intersection samples
        if intersection_only is True:
            is_intersection = r.get("intersection_exist", False)
            if not is_intersection:
                continue

        filtered_scores.append(r["score"])

    if not filtered_scores:
        return 0.0

    avg = float(np.mean(filtered_scores))

    # Log filter info
    lang_str = language if language else "all"
    subset_str = "intersection" if intersection_only else "fullset"
    eval_logger.debug(f"Aggregating {lang_str} {subset_str}: {avg:.4f} (n={len(filtered_scores)})")

    return avg


# ============================================
# English - Fullset Aggregations
# ============================================


def gedit_bench_aggregate_en_fullset_semantics(results):
    """Aggregate English fullset semantics scores"""
    return _aggregate_by_filter(results, language="en", intersection_only=None)


def gedit_bench_aggregate_en_fullset_quality(results):
    """Aggregate English fullset quality scores"""
    return _aggregate_by_filter(results, language="en", intersection_only=None)


def gedit_bench_aggregate_en_fullset_overall(results):
    """Aggregate English fullset overall scores"""
    return _aggregate_by_filter(results, language="en", intersection_only=None)


# ============================================
# English - Intersection Aggregations
# ============================================


def gedit_bench_aggregate_en_intersection_semantics(results):
    """Aggregate English intersection semantics scores"""
    return _aggregate_by_filter(results, language="en", intersection_only=True)


def gedit_bench_aggregate_en_intersection_quality(results):
    """Aggregate English intersection quality scores"""
    return _aggregate_by_filter(results, language="en", intersection_only=True)


def gedit_bench_aggregate_en_intersection_overall(results):
    """Aggregate English intersection overall scores"""
    return _aggregate_by_filter(results, language="en", intersection_only=True)


# ============================================
# Chinese - Fullset Aggregations
# ============================================


def gedit_bench_aggregate_cn_fullset_semantics(results):
    """Aggregate Chinese fullset semantics scores"""
    return _aggregate_by_filter(results, language="cn", intersection_only=None)


def gedit_bench_aggregate_cn_fullset_quality(results):
    """Aggregate Chinese fullset quality scores"""
    return _aggregate_by_filter(results, language="cn", intersection_only=None)


def gedit_bench_aggregate_cn_fullset_overall(results):
    """Aggregate Chinese fullset overall scores"""
    return _aggregate_by_filter(results, language="cn", intersection_only=None)


# ============================================
# Chinese - Intersection Aggregations
# ============================================


def gedit_bench_aggregate_cn_intersection_semantics(results):
    """Aggregate Chinese intersection semantics scores"""
    return _aggregate_by_filter(results, language="cn", intersection_only=True)


def gedit_bench_aggregate_cn_intersection_quality(results):
    """Aggregate Chinese intersection quality scores"""
    return _aggregate_by_filter(results, language="cn", intersection_only=True)


def gedit_bench_aggregate_cn_intersection_overall(results):
    """Aggregate Chinese intersection overall scores"""
    return _aggregate_by_filter(results, language="cn", intersection_only=True)
