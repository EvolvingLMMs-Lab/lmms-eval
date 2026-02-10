# OSI-Bench: A Benchmark for Object-centric Spatial Intelligence
# https://huggingface.co/datasets/HarmlessSR07/OSI-Bench
# Adapted from VLMEvalKit implementation

import os
import re
from functools import partial
from pathlib import Path

import numpy as np
import yaml
from loguru import logger as eval_logger

# Question categories
# Numerical Answer (NA) categories
NA_CATEGORIES = [
    "absolute_distance",
    "relative_direction_angular",
    "trajectory_length",
    "absolute_speed",
    "absolute_displacement",
    "object_3d_localization",
    "depth_aware_counting",
]

# Multiple Choice Answer (MCQ) categories
MCQ_CATEGORIES = [
    "relative_distance",
    "relative_direction_categorical",
    "relative_direction_categorical_cardinal",
    "relative_direction_categorical_ordinal",
    "trajectory_description",
]

# Categories requiring special MRA thresholds
SPEED_DISPLACEMENT_CATEGORIES = ["absolute_speed", "absolute_displacement"]
TRAJECTORY_LENGTH_CATEGORIES = ["trajectory_length"]

# Get cache directory from config
hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)

with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for line in raw_data:
        if "!function" not in line:
            safe_data.append(line)
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]
cache_dir = os.path.join(base_cache_dir, cache_name)

eval_logger.info(f"[osi_bench] cache_dir = {cache_dir}")


##################
# Answer extraction helpers
##################
def extract_number_from_prediction(text):
    """Extract the last numeric value from prediction text."""
    text = str(text).replace(",", "")
    all_numbers = re.findall(r"[+-]?\d+(?:\.\d+)?", text)
    return all_numbers[-1] if all_numbers else None


def extract_option_from_prediction(text):
    """Extract MCQ option letter (A-Z) from prediction text."""
    text = str(text)

    # Try pattern: 答案是 A or Answer: A
    match = re.search(r'[\(\[A-Z]\s*答案是?[:：]?\s*\'?"?([A-Z])\'?"?', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Try pattern: [A] or (A) or space-separated A
    match = re.search(r"[\[\(\s,.]([A-Z])[\]\)\s,.]", f" {text} ")
    if match:
        return match.group(1).upper()

    # Try pattern: starts with A
    match = re.match(r"\s*([A-Z])", text)
    if match:
        return match.group(1).upper()

    return None


##################
# MRA (Mean Relative Accuracy) calculation
##################
def calculate_mra(prediction_text, answer_text, start=0.5, end=0.95, interval=0.05):
    """
    Calculate Mean Relative Accuracy (MRA) score for numerical questions.
    """
    pred_num_str = extract_number_from_prediction(prediction_text)

    if pred_num_str is None:
        return 0.0

    try:
        pred = float(pred_num_str)
        ans = float(answer_text)
    except (ValueError, TypeError):
        return 0.0

    if ans == 0:
        return 1.0 if pred == 0 else 0.0

    relative_error = abs(pred - ans) / abs(ans)

    num_pts = int((end - start) / interval + 2)
    conf_intervs = np.linspace(start, end, num_pts)
    error_tolerances = 1.0 - conf_intervs
    passed_checks = relative_error <= error_tolerances

    score = np.mean(passed_checks)
    return float(score)


def calculate_mra_with_threshold(prediction_text, answer_text, threshold=0.30, start=0.5, end=0.95, interval=0.05):
    """
    Calculate MRA with special handling for zero ground truth.

    Matches VLMEvalKit logic:
    - If ans == 0 and pred < threshold: return 1.0 (both considered stationary/zero)
    - If ans == 0 and pred >= threshold: compute MRA with threshold as reference
    - Otherwise: compute standard MRA

    Used for speed, displacement, and trajectory_length categories.
    """
    pred_num_str = extract_number_from_prediction(prediction_text)

    if pred_num_str is None:
        return 0.0

    try:
        pred = float(pred_num_str)
        ans = float(answer_text)
    except (ValueError, TypeError):
        return 0.0

    # Special handling ONLY when ground truth is exactly 0
    # (matches VLMEvalKit _apply_osi_na_mra logic)
    if ans == 0:
        if pred < threshold:
            return 1.0
        else:
            # Use threshold as reference for MRA calculation
            ans = threshold

    relative_error = abs(pred - ans) / abs(ans)

    num_pts = int((end - start) / interval + 2)
    conf_intervs = np.linspace(start, end, num_pts)
    error_tolerances = 1.0 - conf_intervs
    passed_checks = relative_error <= error_tolerances

    score = np.mean(passed_checks)
    return float(score)


##################
# Prompt building
##################
def build_prompt(doc, lmms_eval_specific_kwargs=None, include_video_length=False):
    """Build the prompt text based on question category."""
    question_text = doc["question"]
    category = doc.get("category", "unknown")
    video_length = doc.get("video_length", 0)
    options = doc.get("options", [])

    # Preamble for numeric-tagged objects
    preamble_num_tagged = "These are frames of a video.\n" "In the video, objects are identified by numeric tags shown nearby.\n" "With that in mind, please answer the following question based on the video."

    prompt_text = ""

    # Numerical answer categories
    if category in ["absolute_distance", "relative_direction_angular", "trajectory_length"]:
        instruction = "Your answer must be only the final numeric value, without units or any other text."
        prompt_text = f"{preamble_num_tagged}\nQuestion: {question_text}\n\n{instruction}\n"

    # Numerical answer categories that benefit from video length info
    elif category in ["absolute_speed", "absolute_displacement", "object_3d_localization", "depth_aware_counting"]:
        instruction = "Your answer must be only the final numeric value, without units or any other text."
        prompt_text = f"{preamble_num_tagged}\nQuestion: {question_text}\n\n{instruction}"

    # MCQ categories
    elif category in ["relative_distance", "relative_direction_categorical", "relative_direction_categorical_cardinal", "relative_direction_categorical_ordinal"]:
        instruction = "Your answer must be only the single letter (e.g., A, B, C, or D) of the correct option."
        options_text = "\n".join(options) if options else ""
        prompt_text = f"{preamble_num_tagged}\nQuestion: {question_text}\n{options_text}\n\n{instruction}"

    # trajectory_description - qualitative ego-motion, no numeric tags needed
    elif category == "trajectory_description":
        instruction = "Your answer must be only the single letter (e.g., A, B, C, or D) of the correct option."
        options_text = "\n".join(options) if options else ""
        prompt_text = f"Question: {question_text}\n{options_text}\n\n{instruction}"

    else:
        # Fallback for unknown categories
        eval_logger.warning(f"[osi_bench] Unknown category '{category}'. Using generic prompt.")
        instruction = "Your answer must be only the final numeric value, without units or any other text."
        prompt_text = f"{preamble_num_tagged}\nQuestion: {question_text}\n\n{instruction}\n"

    prompt_text = prompt_text + "\nThe answer is:"

    # Add post_prompt if provided
    if lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs.get("default", {}).get("post_prompt", "")
        if post_prompt:
            prompt_text += " " + post_prompt

    return prompt_text


##################
# doc_to_visual functions
##################
def osi_bench_doc_to_visual_video(doc):
    """Return video path for native video input mode."""
    video_file = doc.get("file_name", doc.get("video", "") + ".mp4")
    video_path = os.path.join(cache_dir, "videos", video_file)

    if not os.path.exists(video_path):
        # Try alternative path structure
        video_path = os.path.join(cache_dir, video_file)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    return [video_path]


def osi_bench_doc_to_visual_frames(doc, lmms_eval_specific_kwargs=None):
    """
    Return video frames as PIL Images for frame-based input mode.
    """
    from decord import VideoReader, cpu
    from PIL import Image

    video_file = doc.get("file_name", doc.get("video", "") + ".mp4")
    video_path = os.path.join(cache_dir, "videos", video_file)

    if not os.path.exists(video_path):
        video_path = os.path.join(cache_dir, video_file)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Get number of frames from config
    num_frames = 32
    if lmms_eval_specific_kwargs:
        num_frames = lmms_eval_specific_kwargs.get("default", {}).get("num_frames", 32)

    # Load video and sample frames uniformly
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)

    # Handle case where video has fewer frames than requested
    if total_frames <= num_frames:
        # Use all available frames
        indices = np.arange(total_frames)
        eval_logger.debug(f"[osi_bench] Video has only {total_frames} frames, using all of them (requested {num_frames})")
    else:
        # Sample uniformly
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = vr.get_batch(indices).asnumpy()

    # Convert to PIL Images
    pil_images = [Image.fromarray(frame) for frame in frames]

    eval_logger.debug(f"[osi_bench] Loaded {len(pil_images)} frames from {video_path}")

    return pil_images


##################
# doc_to_text functions
##################
def osi_bench_doc_to_text_video(doc, lmms_eval_specific_kwargs=None, include_visual_token=True):
    """
    Build text prompt for native video input mode.

    Args:
        doc: Document containing question data
        lmms_eval_specific_kwargs: Config kwargs
        include_visual_token: If True, append <video> token for simple models.
                              Set to False when using doc_to_messages (chat models).

    Order controlled by 'visual_first' in lmms_eval_specific_kwargs:
    - visual_first=False (default): [text prompt] → [<video>]
    - visual_first=True: [<video>] → [text prompt]
    """
    prompt = build_prompt(doc, lmms_eval_specific_kwargs)

    # Check visual_first option
    visual_first = False
    if lmms_eval_specific_kwargs:
        visual_first = lmms_eval_specific_kwargs.get("default", {}).get("visual_first", False)

    # Append <video> token for simple models that don't use doc_to_messages
    if include_visual_token:
        if visual_first:
            # Video token first, then text prompt
            prompt = "<video>\n" + prompt
        else:
            # Text prompt first, then video token
            prompt = prompt + "\n<video>"

    return prompt


def osi_bench_doc_to_text_frames(doc, lmms_eval_specific_kwargs=None, include_visual_token=True):
    """
    Build text prompt for frame-based input mode.

    Args:
        doc: Document containing question data
        lmms_eval_specific_kwargs: Config kwargs including num_frames
        include_visual_token: If True, append <image> tokens for simple models.
                              Set to False when using doc_to_messages (chat models).

    Order controlled by 'visual_first' in lmms_eval_specific_kwargs:
    - visual_first=False (default): [text prompt with video_length] → [<image> tokens]
    - visual_first=True (VLMEvalKit format): [time_context] → [<image> tokens] → [question prompt]
    """
    video_length = doc.get("video_length") or 0  # Handle None
    num_frames = 32
    if lmms_eval_specific_kwargs:
        num_frames = lmms_eval_specific_kwargs.get("default", {}).get("num_frames", 32)

    # Check visual_first option
    visual_first = False
    if lmms_eval_specific_kwargs:
        visual_first = lmms_eval_specific_kwargs.get("default", {}).get("visual_first", False)
    eval_logger.debug(f"[osi_bench] visual_first = {visual_first}")

    if visual_first:
        # VLMEvalKit format: [time_context] → [<image> tokens] → [question prompt]
        parts = []

        # 1. Add time context first (if video_length available)
        if video_length and video_length > 0:
            time_context = f"The video is {round(video_length, 2)} seconds long. " f"The following {num_frames} frames are uniformly sampled from it " "in chronological order:"
            parts.append(time_context)

        # 2. Add image tokens
        if include_visual_token:
            image_tokens = "<image>" * num_frames
            parts.append(image_tokens)

        # 3. Add question prompt last
        question_prompt = build_prompt(doc, lmms_eval_specific_kwargs)
        parts.append(question_prompt)

        prompt = "\n".join(parts)
    else:
        # Default format: [text prompt with video_length] → [<image> tokens]
        prompt = build_prompt(doc, lmms_eval_specific_kwargs)

        # Append frame context AFTER main prompt if video_length is available
        if video_length and video_length > 0:
            frame_context = f"The video is {round(video_length, 2)} seconds long. " f"The following {num_frames} frames are uniformly sampled from it " "in chronological order:"
            prompt = prompt + "\n" + frame_context

        # Append <image> tokens for simple models that don't use doc_to_messages
        if include_visual_token:
            image_tokens = "\n" + "<image>" * num_frames
            prompt = prompt + image_tokens

    return prompt


##################
# doc_to_messages functions (for chat models with interleaved support)
##################
def osi_bench_doc_to_messages_video(doc, lmms_eval_specific_kwargs=None):
    """
    Build messages for native video mode.

    Order controlled by 'visual_first' in lmms_eval_specific_kwargs:
    - visual_first=False (default): [text prompt] → [video]
    - visual_first=True: [video] → [text prompt]
    """
    # Set include_visual_token=False since messages handle visuals separately
    prompt = osi_bench_doc_to_text_video(doc, lmms_eval_specific_kwargs, include_visual_token=False)
    video_paths = osi_bench_doc_to_visual_video(doc)

    # Check visual_first option
    visual_first = False
    if lmms_eval_specific_kwargs:
        visual_first = lmms_eval_specific_kwargs.get("default", {}).get("visual_first", False)

    messages = [{"role": "user", "content": []}]

    if visual_first:
        # Add video(s) first
        for video_path in video_paths:
            messages[0]["content"].append({"type": "video", "url": video_path})
        # Add text prompt
        messages[0]["content"].append({"type": "text", "text": prompt})
    else:
        # Add text prompt first
        messages[0]["content"].append({"type": "text", "text": prompt})
        # Add video(s)
        for video_path in video_paths:
            messages[0]["content"].append({"type": "video", "url": video_path})

    return messages


def osi_bench_doc_to_messages_frames(doc, lmms_eval_specific_kwargs=None):
    """
    Build messages for frame-based mode.

    Order controlled by 'visual_first' in lmms_eval_specific_kwargs:
    - visual_first=False (default): [text prompt with video_length] → [frames]
    - visual_first=True (VLMEvalKit format): [time_context] → [frames] → [question prompt]
    """
    pil_images = osi_bench_doc_to_visual_frames(doc, lmms_eval_specific_kwargs)

    # Check visual_first option
    visual_first = False
    if lmms_eval_specific_kwargs:
        visual_first = lmms_eval_specific_kwargs.get("default", {}).get("visual_first", False)

    messages = [{"role": "user", "content": []}]

    if visual_first:
        # VLMEvalKit format: [time_context] → [frames] → [question prompt]
        video_length = doc.get("video_length") or 0

        actual_num_frames = len(pil_images)

        # 1. Add time context prompt first (if video_length available)
        if video_length and video_length > 0:
            time_context_prompt = f"The video is {round(video_length, 2)} seconds long. " f"The following {actual_num_frames} frames are uniformly sampled from it " "in chronological order:"
            messages[0]["content"].append({"type": "text", "text": time_context_prompt})

        # 2. Add frames
        for img in pil_images:
            messages[0]["content"].append({"type": "image", "url": img})

        # 3. Add question prompt last (without video_length context since it's already added)
        question_prompt = build_prompt(doc, lmms_eval_specific_kwargs)
        messages[0]["content"].append({"type": "text", "text": question_prompt})
    else:
        # Default format: [text prompt with video_length] → [frames]
        prompt = osi_bench_doc_to_text_frames(doc, lmms_eval_specific_kwargs, include_visual_token=False)
        messages[0]["content"].append({"type": "text", "text": prompt})
        for img in pil_images:
            messages[0]["content"].append({"type": "image", "url": img})

    return messages


# All metric keys (must match YAML metric_list)
NA_METRIC_KEYS = [f"{cat}_mra" for cat in NA_CATEGORIES]
MCQ_METRIC_KEYS = [f"{cat}_acc" for cat in MCQ_CATEGORIES]
ALL_METRIC_KEYS = NA_METRIC_KEYS + MCQ_METRIC_KEYS

# Categories for relative_direction_avg (average of 4 direction metrics)
RELATIVE_DIRECTION_CATEGORIES = [
    "relative_direction_angular",
    "relative_direction_categorical",
    "relative_direction_categorical_cardinal",
    "relative_direction_categorical_ordinal",
]


##################
# Result processing
##################
def osi_bench_process_results(doc, results):
    """
    Process model results and compute scores.
    Returns metrics for overall and per-category accuracy.
    Each category has its own metric key for separate aggregation.
    """
    prediction = results[0].strip() if results else ""
    answer = doc.get("answer", "")
    question_type = doc.get("question_type", "")
    category = doc.get("category", "unknown")

    score = 0.0

    if question_type == "numerical" or category in NA_CATEGORIES:
        # Numerical answer - use MRA
        if category in SPEED_DISPLACEMENT_CATEGORIES:
            score = calculate_mra_with_threshold(prediction, answer, threshold=0.30)
        elif category in TRAJECTORY_LENGTH_CATEGORIES:
            score = calculate_mra_with_threshold(prediction, answer, threshold=2.0)
        else:
            score = calculate_mra(prediction, answer)

    elif question_type == "mcq" or category in MCQ_CATEGORIES:
        # MCQ - exact match on option letter
        pred_opt = extract_option_from_prediction(prediction)
        score = 1.0 if pred_opt == str(answer).strip().upper() else 0.0

    else:
        eval_logger.warning(f"[osi_bench] Unknown question_type '{question_type}' for category '{category}'")

    # Build result dict with each metric key separately
    # Use {"score": X, "total": 1} for matching category, {"score": 0, "total": 0} for non-matching
    result_dict = {}

    # Overall metric - always count this sample
    result_dict["overall"] = {"score": score, "total": 1}

    # Determine the metric key for this sample's category
    if category in NA_CATEGORIES:
        current_metric_key = f"{category}_mra"
    elif category in MCQ_CATEGORIES:
        current_metric_key = f"{category}_acc"
    else:
        current_metric_key = None

    # Per-category metrics - only count if sample belongs to that category
    for metric_key in ALL_METRIC_KEYS:
        if metric_key == current_metric_key:
            result_dict[metric_key] = {"score": score, "total": 1}
        else:
            result_dict[metric_key] = {"score": 0, "total": 0}

    # Aggregate metric: relative_direction_avg (averages 4 direction categories)
    if category in RELATIVE_DIRECTION_CATEGORIES:
        result_dict["relative_direction_avg"] = {"score": score, "total": 1}
    else:
        result_dict["relative_direction_avg"] = {"score": 0, "total": 0}

    return result_dict


##################
# Aggregation functions
##################
def _aggregate_category(results):
    """Aggregate results for a category. Sums scores and totals."""
    total_score = 0.0
    total_count = 0

    for result in results:
        total_score += result.get("score", 0)
        total_count += result.get("total", 0)

    if total_count == 0:
        return 0.0

    accuracy = (total_score / total_count) * 100
    return round(accuracy, 2)


def aggregate_overall(results):
    """Aggregate overall score across all samples."""
    return _aggregate_category(results)


# --- NA category aggregation functions (MRA) ---
def aggregate_absolute_distance_mra(results):
    """Aggregate MRA for absolute_distance category."""
    return _aggregate_category(results)


def aggregate_relative_direction_angular_mra(results):
    """Aggregate MRA for relative_direction_angular category."""
    return _aggregate_category(results)


def aggregate_trajectory_length_mra(results):
    """Aggregate MRA for trajectory_length category."""
    return _aggregate_category(results)


def aggregate_absolute_speed_mra(results):
    """Aggregate MRA for absolute_speed category."""
    return _aggregate_category(results)


def aggregate_absolute_displacement_mra(results):
    """Aggregate MRA for absolute_displacement category."""
    return _aggregate_category(results)


def aggregate_object_3d_localization_mra(results):
    """Aggregate MRA for object_3d_localization category."""
    return _aggregate_category(results)


def aggregate_depth_aware_counting_mra(results):
    """Aggregate MRA for depth_aware_counting category."""
    return _aggregate_category(results)


# --- MCQ category aggregation functions (accuracy) ---
def aggregate_relative_distance_acc(results):
    """Aggregate accuracy for relative_distance category."""
    return _aggregate_category(results)


def aggregate_relative_direction_categorical_acc(results):
    """Aggregate accuracy for relative_direction_categorical category."""
    return _aggregate_category(results)


def aggregate_relative_direction_categorical_cardinal_acc(results):
    """Aggregate accuracy for relative_direction_categorical_cardinal category."""
    return _aggregate_category(results)


def aggregate_relative_direction_categorical_ordinal_acc(results):
    """Aggregate accuracy for relative_direction_categorical_ordinal category."""
    return _aggregate_category(results)


def aggregate_trajectory_description_acc(results):
    """Aggregate accuracy for trajectory_description category."""
    return _aggregate_category(results)


# --- Aggregate metrics (averaging multiple categories) ---
def aggregate_relative_direction_avg(results):
    """Aggregate average across all 4 relative direction categories."""
    return _aggregate_category(results)
