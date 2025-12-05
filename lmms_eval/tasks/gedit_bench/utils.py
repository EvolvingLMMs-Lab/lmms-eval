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
try:
    from lmms_eval.tasks.gedit_bench.viescore import VIEScore

    VIESCORE_AVAILABLE = True
except ImportError:
    VIESCORE_AVAILABLE = False
    eval_logger.warning("VIEScore not available. Please install it: pip install viescore. " "Evaluation scores will not be computed.")

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


def _get_vie_score(backbone: str = "gpt4o", key_path: Optional[str] = None):
    """
    Get or create VIEScore instance.
    Note: In multi-process environments, each process will have its own instance.
    """
    if not VIESCORE_AVAILABLE:
        raise ImportError("VIEScore is not available. Please install it: pip install viescore")

    # Create a new instance each time (safe for multi-process)
    # VIEScore initialization is relatively lightweight
    return VIEScore(backbone=backbone, task="tie", key_path=key_path)


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


def _save_image_to_structure(
    image_path: str,
    key: str,
    task_type: str,
    instruction_language: str,
    output_base_dir: str,
    model_name: str = "default",
) -> str:
    """
    Save image to the required directory structure:
    results/{model_name}/fullset/{task_type}/{instruction_language}/{key}.png

    Args:
        image_path: Path to the generated image
        key: Unique key for this sample
        task_type: Task type (e.g., "background_change")
        instruction_language: Language of instruction ("en" or "cn")
        output_base_dir: Base directory for outputs
        model_name: Name of the model being evaluated

    Returns:
        Path to the saved image
    """
    # Create directory structure
    save_dir = os.path.join(output_base_dir, model_name, "fullset", task_type, instruction_language)
    os.makedirs(save_dir, exist_ok=True)

    # Save image with key as filename (preserve extension if exists)
    if os.path.exists(image_path):
        # Copy image to new location
        save_path = os.path.join(save_dir, f"{key}.png")
        shutil.copy2(image_path, save_path)
        return save_path
    else:
        eval_logger.warning(f"Image not found at {image_path}, skipping save")
        return ""


def _create_result_entry(key, task_type, instruction_language, score, intersection_exist):
    """Helper to create a result entry dict"""
    return {
        "key": key,
        "task_type": task_type,
        "instruction_language": instruction_language,
        "score": score,
        "intersection_exist": intersection_exist,
    }


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
    1. Parse JSON output to extract text and images
    2. Save images to required directory structure
    3. Evaluate using VIEScore

    Args:
        doc: Document containing input image, instruction, key, task_type, etc.
        results: Model predictions [JSON string with {"text": "...", "images": [...]}]
        **kwargs: Additional arguments (may include full_docs)

    Returns:
        Dict with metrics for all breakdown categories
    """
    # Get configuration from environment variables or use defaults
    # Note: defaults should match bagel.py's output_image_dir structure
    model_name = os.getenv("GEDIT_BENCH_MODEL_NAME", "bagel")
    output_base_dir = os.getenv("GEDIT_BENCH_OUTPUT_DIR", "./logs/bagel_persistent_folder/bagel_generated_images")
    vie_backbone = os.getenv("GEDIT_BENCH_VIE_BACKBONE", "gpt4o")
    vie_key_path = os.getenv("GEDIT_BENCH_VIE_KEY_PATH", None)
    pred = results[0] if results else "{}"
    try:
        pred = json.loads(pred)
    except (json.JSONDecodeError, TypeError):
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

    # Get input image (try different possible field names)
    input_image = doc.get("input_image") or doc.get("input_image_raw")

    # If input_image is None, try to load from saved _SRCIMG file
    # This happens when process_results is called after generation, and the doc doesn't contain PIL image data
    input_image_pil = None
    if input_image is not None:
        input_image_pil = input_image.convert("RGB") if hasattr(input_image, "convert") else input_image
    else:
        # Try to load from _SRCIMG file saved during generation
        src_img_path = os.path.join(output_base_dir, model_name, "fullset", task_type, instruction_language, f"{key}_SRCIMG.png")
        if os.path.exists(src_img_path):
            try:
                input_image_pil = Image.open(src_img_path).convert("RGB")
                eval_logger.debug(f"Loaded source image from {src_img_path}")
            except Exception as e:
                eval_logger.warning(f"Failed to load source image from {src_img_path}: {e}")

    if input_image_pil is None:
        eval_logger.warning(f"No input image found for key {key} (neither in doc nor as _SRCIMG file)")
        return _create_all_metric_results(key, task_type, instruction_language, 0.0, 0.0, 0.0, intersection_exist)

    # Save generated images to required structure (or use existing)
    edited_image_path = None
    if model_images and len(model_images) > 0:
        # Use first generated image
        generated_image_path = model_images[0]
        # Check if the image is already at the target location
        if os.path.exists(generated_image_path):
            edited_image_path = generated_image_path
            # Also copy to standard structure if not already there
            target_path = os.path.join(output_base_dir, model_name, "fullset", task_type, instruction_language, f"{key}.png")
            if generated_image_path != target_path and not os.path.exists(target_path):
                edited_image_path = _save_image_to_structure(
                    generated_image_path,
                    key,
                    task_type,
                    instruction_language,
                    output_base_dir,
                    model_name,
                )
        else:
            eval_logger.warning(f"Generated image not found at {generated_image_path}")

    # If no image from model results, try to find existing generated image in the standard location
    if edited_image_path is None:
        existing_path = os.path.join(output_base_dir, model_name, "fullset", task_type, instruction_language, f"{key}.png")
        if os.path.exists(existing_path):
            edited_image_path = existing_path
            eval_logger.debug(f"Found existing generated image at {existing_path}")

    # If still no edited image, return zero scores
    if edited_image_path is None:
        eval_logger.warning(f"No generated images found for key {key}")
        return _create_all_metric_results(key, task_type, instruction_language, 0.0, 0.0, 0.0, intersection_exist)

    # Evaluate using VIEScore
    if not VIESCORE_AVAILABLE:
        eval_logger.warning("VIEScore not available, skipping evaluation")
        return _create_all_metric_results(key, task_type, instruction_language, 0.0, 0.0, 0.0, intersection_exist)

    try:
        # Load edited image
        edited_image_pil = Image.open(edited_image_path).convert("RGB")

        # Resize images to target area (512x512 equivalent)
        source_img_width, source_img_height, _ = calculate_dimensions(512 * 512, input_image_pil.width / input_image_pil.height)
        edited_img_width, edited_img_height, _ = calculate_dimensions(512 * 512, edited_image_pil.width / edited_image_pil.height)

        input_image_pil = input_image_pil.resize((source_img_width, source_img_height))
        edited_image_pil = edited_image_pil.resize((edited_img_width, edited_img_height))

        # Get VIEScore instance
        vie_score = _get_vie_score(backbone=vie_backbone, key_path=vie_key_path)

        # Evaluate: VIEScore.evaluate returns [semantics_score, quality_score, overall_score]
        score_list = vie_score.evaluate([input_image_pil, edited_image_pil], instruction)
        semantics_score, quality_score, overall_score = score_list

        eval_logger.info(f"[{task_type}] Key {key}: " f"Semantics={semantics_score:.3f}, " f"Quality={quality_score:.3f}, " f"Overall={overall_score:.3f}, " f"Language={instruction_language}, " f"Intersection={intersection_exist}")

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


# ============================================
# Legacy Functions (for backward compatibility)
# ============================================


def gedit_bench_aggregate_en_fullset(results):
    """Aggregate English fullset scores (all English samples)"""
    return _aggregate_by_filter(results, language="en", intersection_only=None)


def gedit_bench_aggregate_en_intersection(results):
    """Aggregate English intersection subset scores"""
    return _aggregate_by_filter(results, language="en", intersection_only=True)


def gedit_bench_aggregate_cn_fullset(results):
    """Aggregate Chinese fullset scores (all Chinese samples)"""
    return _aggregate_by_filter(results, language="cn", intersection_only=None)


def gedit_bench_aggregate_cn_intersection(results):
    """Aggregate Chinese intersection subset scores"""
    return _aggregate_by_filter(results, language="cn", intersection_only=True)


def gedit_bench_aggregate_intersection(results):
    """Aggregate intersection subset scores (all languages)"""
    return _aggregate_by_filter(results, language=None, intersection_only=True)


def gedit_bench_aggregate_fullset(results):
    """Aggregate fullset scores (all samples, all languages)"""
    return _aggregate_by_filter(results, language=None, intersection_only=None)
