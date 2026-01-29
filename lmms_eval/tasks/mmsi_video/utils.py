# MMSI-Video-Bench: A Holistic Benchmark for Video-Based Spatial Intelligence
# https://huggingface.co/datasets/rbler/MMSI-Video-Bench

import json
import os
import re
from collections import defaultdict
from pathlib import Path

import datasets
import yaml
from loguru import logger as eval_logger
from PIL import Image

# Get cache directory following the lmms-eval pattern (like vsibench/sitebench)
# lmms-eval automatically downloads the dataset and extracts zip files to $HF_HOME/<cache_dir>/
hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)

# Read cache_dir from YAML config
with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for line in raw_data:
        if "!function" not in line:
            safe_data.append(line)
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]
cache_dir = os.path.join(base_cache_dir, cache_name)
eval_logger.info(f"[mmsi_video] cache_dir = {cache_dir}")


# Main benchmark categories
MAIN_CATEGORIES = {
    "(Cross-Video) Memory Update",
    "(Cross-Video) Multi-View Integration",
    "Planning",
    "Prediction",
    "(Motion Understanding) Camera Motion",
    "(Motion Understanding) Instance Motion",
    "(Motion Understanding) Interactive Motion",
    "(Spatial Construction) Instance-Instance Spatial Relationship",
    "(Spatial Construction) Instance-Scene Spatial Relationship",
    "(Spatial Construction) Scene-Scene Spatial Relationship",
    "(Spatial Construction) Instance/Scene Attribute",
    "(Spatial Construction) Camera-Instance Spatial Relationship",
    "(Spatial Construction) Camera-Scene Spatial Relationship",
}

# Sub-benchmarks for aggregation
SUB_BENCHS = ["Easy2hard", "Grounding", "Indoor_Scene_Perception", "Robot"]


##################
# Sampling helper functions from official loader
##################
def interval_sampling_list(a: int, b: int) -> list[int]:
    """Sample b indices uniformly from a total of a items."""
    if b <= 0:
        return []
    if b == 1:
        return [0]
    step = (a - 1) / (b - 1)
    indices = [int(i * step) for i in range(b)]
    return indices


def proportional_sample_from_lists(num_list: list[int], k: int) -> list[list[int]]:
    """
    Proportionally sample k total indices across multiple lists.
    Returns list of index lists for each original list.
    """
    total_indices = interval_sampling_list(sum(num_list), k)

    sum_list = [sum(num_list[: i + 1]) for i in range(len(num_list))]
    sum_list = [0] + sum_list

    indices_list = []
    for i in range(len(sum_list) - 1):
        raw_indices = [idx - sum_list[i] for idx in total_indices if sum_list[i] <= idx < sum_list[i + 1]]
        indices_list.append(raw_indices)
    return indices_list


##################
# Answer extraction (from VLMEvalKit)
##################
def clear_words(text: str) -> str:
    """Remove common noise characters from text."""
    return text.replace(" ", "").replace('"', "").replace("'", "").replace("\n", "").replace(":", "")


def extract_answer(response: str) -> str:
    """
    Extract MCQ answer (A-F) from model response.
    Uses multiple fallback patterns similar to VLMEvalKit.
    """
    if response is None or "no answer" in response.lower():
        return "O"  # Invalid/no answer marker

    response = response.replace("<answer>", "").replace("</answer>", "")

    # Try boxed format: \boxed{A}
    if "boxed{" in response:
        split_text = response.split("boxed{")[1].split("}")[0]
        split_text = clear_words(split_text)
        if split_text in ["A", "B", "C", "D", "E", "F"]:
            return split_text

    # Try various answer patterns
    patterns = [
        '"answer":',
        "answer is",
        "answer:",
        '"Answer":',
        "Answer is",
        "Answer:",
        "The answer is",
        "the answer is",
    ]
    for pattern in patterns:
        if pattern in response:
            split_text = response.split(pattern)[-1]
            split_text = split_text.split(",")[0].split(".")[0]
            split_text = clear_words(split_text)
            if split_text in ["A", "B", "C", "D", "E", "F"]:
                return split_text

    # Try first word if it's a valid choice
    first_word = clear_words(response.split(".")[0])
    if first_word in ["A", "B", "C", "D", "E", "F"]:
        return first_word

    # Fallback: look for any choice letter in the response
    for choice in ["A", "B", "C", "D", "E", "F"]:
        if f"({choice})" in response or f" {choice} " in f" {response} " or f"{choice}." in response:
            return choice

    return "O"  # No valid answer found


def is_nan_or_none(value) -> bool:
    """Check if value is None, NaN, or empty."""
    if value is None:
        return True
    try:
        if isinstance(value, str) and value.lower() in ["nan", "null", "none", ""]:
            return True
        if isinstance(value, float) and value != value:  # nan != nan
            return True
    except Exception:
        pass
    return False


##################
# doc_to_visual functions
##################
def doc_to_visual_video(doc):
    """
    Return video paths for native video input mode.
    Returns list of video file paths from video_list.
    """
    video_paths = []
    video_list = doc.get("video_list", [])
    for video_info in video_list:
        video_path = os.path.join(cache_dir, "videos", video_info["path"])
        video_paths.append(video_path)

    # Also return ref_images as PIL Images for mixed content
    ref_images = doc.get("ref_images", [])
    ref_image_pils = []
    for ref_img_path in ref_images:
        full_path = os.path.join(cache_dir, "ref_images", ref_img_path)
        ref_image_pils.append(Image.open(full_path).convert("RGB"))

    # For native video mode, we return videos first, then ref images
    # The doc_to_messages function will handle proper interleaving
    return video_paths + ref_image_pils


def doc_to_visual_frames(doc, lmms_eval_specific_kwargs=None):
    """
    Return sampled frames as PIL Images for frame-based input mode.
    Respects max_frame parameter from lmms_eval_specific_kwargs.
    """

    max_frame = 50  # default
    if lmms_eval_specific_kwargs:
        max_frame = lmms_eval_specific_kwargs.get("default", {}).get("max_frame", 50)

    frames_list = doc.get("frames_list", [])

    if isinstance(frames_list, str):
        import ast

        frames_list = ast.literal_eval(frames_list)

    # Calculate total frames and sample proportionally
    total_frames = sum(len(frames) for frames in frames_list)
    eval_logger.debug(f"[doc_to_visual_frames] total_frames: {total_frames}, num_video_segments: {len(frames_list)}")

    sampled_frames_list = []

    if total_frames > max_frame:
        indices_list = proportional_sample_from_lists([len(frames) for frames in frames_list], max_frame)
        for i, indices in enumerate(indices_list):
            if len(indices) < 1:
                indices = [0]  # Ensure at least one frame per video
            sampled_frames_list.append([frames_list[i][j] for j in indices])
    else:
        sampled_frames_list = frames_list

    # Load sampled frames as PIL Images
    pil_images = []
    eval_logger.debug(f"[doc_to_visual_frames] sampled_frames_list has {len(sampled_frames_list)} segments")
    for seg_idx, frames in enumerate(sampled_frames_list):
        for frame_path in frames:
            full_path = os.path.join(cache_dir, "frames", frame_path)
            if not os.path.exists(full_path):
                eval_logger.error(f"[doc_to_visual_frames] Frame not found: {full_path}")
                raise FileNotFoundError(f"Frame not found: {full_path}")
            pil_images.append(Image.open(full_path).convert("RGB"))

    # Append reference images
    ref_images = doc.get("ref_images", [])
    if isinstance(ref_images, str):
        import ast

        ref_images = ast.literal_eval(ref_images)

    for ref_img_path in ref_images:
        full_path = os.path.join(cache_dir, "ref_images", ref_img_path)
        if not os.path.exists(full_path):
            eval_logger.error(f"[doc_to_visual_frames] Ref image not found: {full_path}")
            raise FileNotFoundError(f"Ref image not found: {full_path}")
        pil_images.append(Image.open(full_path).convert("RGB"))

    return pil_images


##################
# doc_to_text functions
##################
def _build_prompt_text(doc, lmms_eval_specific_kwargs=None, use_system_prompt=True):
    """Build the full prompt text from document fields."""
    system_prompt = doc.get("system_prompt", "")
    task_prompt = doc.get("task_prompt", "")
    user_prompt = doc.get("user_prompt", "")
    format_prompt = doc.get("format_prompt", "")

    if use_system_prompt and system_prompt:
        prompt = f"{system_prompt}\n{task_prompt}{user_prompt}{format_prompt}"
    else:
        prompt = f"{task_prompt}{user_prompt}{format_prompt}"

    # Add post_prompt if provided
    if lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs.get("default", {}).get("post_prompt", "")
        if post_prompt:
            prompt += post_prompt

    return prompt


def doc_to_text_video(doc, lmms_eval_specific_kwargs=None):
    """
    Build text prompt for native video mode.
    Keeps <video> and <image> placeholders as-is for chat models.
    Simple models will strip these placeholders.
    """
    return _build_prompt_text(doc, lmms_eval_specific_kwargs)


def doc_to_text_frames(doc, lmms_eval_specific_kwargs=None):
    """
    Build text prompt for frame-based mode.
    Replaces <video> with appropriate number of <image> placeholders.
    """

    max_frame = 50
    if lmms_eval_specific_kwargs:
        max_frame = lmms_eval_specific_kwargs.get("default", {}).get("max_frame", 50)

    prompt = _build_prompt_text(doc, lmms_eval_specific_kwargs)

    # Count frames per video segment to replace <video> correctly
    frames_list = doc.get("frames_list", [])
    if isinstance(frames_list, str):
        import ast

        frames_list = ast.literal_eval(frames_list)

    total_frames = sum(len(frames) for frames in frames_list)

    if total_frames > max_frame:
        indices_list = proportional_sample_from_lists([len(frames) for frames in frames_list], max_frame)
        frame_counts = [max(1, len(indices)) for indices in indices_list]
    else:
        frame_counts = [len(frames) for frames in frames_list]

    # Replace each <video> with the corresponding number of <image> tags
    for count in frame_counts:
        prompt = prompt.replace("<video>", "<image>" * count, 1)

    return prompt


##################
# doc_to_messages functions (for chat models)
##################
def doc_to_messages_video(doc, lmms_eval_specific_kwargs=None):
    """
    Build interleaved messages for native video mode.
    Handles mixed <video> and <image> placeholders.
    """
    prompt = _build_prompt_text(doc, lmms_eval_specific_kwargs, use_system_prompt=False)

    # Get video paths and ref images
    video_list = doc.get("video_list", [])
    video_paths = [os.path.join(cache_dir, "videos", v["path"]) for v in video_list]

    ref_images = doc.get("ref_images", [])
    if isinstance(ref_images, str):
        import ast

        ref_images = ast.literal_eval(ref_images)
    ref_image_pils = [Image.open(os.path.join(cache_dir, "ref_images", p)).convert("RGB") for p in ref_images]

    # Parse prompt for <video> and <image> placeholders
    messages = [{"role": "user", "content": []}]

    # Add system prompt as separate text if present
    system_prompt = doc.get("system_prompt", "")
    if system_prompt:
        messages[0]["content"].append({"type": "text", "text": system_prompt})

    # Split by both <video> and <image> placeholders
    pattern = r"(<video>|<image>)"
    parts = re.split(pattern, prompt)

    video_idx = 0
    image_idx = 0

    for part in parts:
        if part == "<video>":
            if video_idx < len(video_paths):
                messages[0]["content"].append({"type": "video", "url": video_paths[video_idx]})
                video_idx += 1
        elif part == "<image>":
            if image_idx < len(ref_image_pils):
                messages[0]["content"].append({"type": "image", "url": ref_image_pils[image_idx]})
                image_idx += 1
        elif part.strip():
            messages[0]["content"].append({"type": "text", "text": part})

    return messages


def doc_to_messages_frames(doc, lmms_eval_specific_kwargs=None):
    """
    Build interleaved messages for frame-based mode.
    Replaces <video> with sampled frames as images.
    """
    max_frame = 50
    if lmms_eval_specific_kwargs:
        max_frame = lmms_eval_specific_kwargs.get("default", {}).get("max_frame", 50)

    prompt = _build_prompt_text(doc, lmms_eval_specific_kwargs, use_system_prompt=False)

    # Get sampled frames
    frames_list = doc.get("frames_list", [])
    if isinstance(frames_list, str):
        import ast

        frames_list = ast.literal_eval(frames_list)

    total_frames = sum(len(frames) for frames in frames_list)
    sampled_frames_list = []

    if total_frames > max_frame:
        indices_list = proportional_sample_from_lists([len(frames) for frames in frames_list], max_frame)
        for i, indices in enumerate(indices_list):
            if len(indices) < 1:
                indices = [0]
            sampled_frames_list.append([frames_list[i][j] for j in indices])
    else:
        sampled_frames_list = frames_list

    # Load frames as PIL Images
    video_frame_lists = []
    for frames in sampled_frames_list:
        pil_frames = []
        for frame_path in frames:
            full_path = os.path.join(cache_dir, "frames", frame_path)
            pil_frames.append(Image.open(full_path).convert("RGB"))
        video_frame_lists.append(pil_frames)

    # Load ref images
    ref_images = doc.get("ref_images", [])
    if isinstance(ref_images, str):
        import ast

        ref_images = ast.literal_eval(ref_images)
    ref_image_pils = [Image.open(os.path.join(cache_dir, "ref_images", p)).convert("RGB") for p in ref_images]

    # Parse prompt for <video> and <image> placeholders
    messages = [{"role": "user", "content": []}]

    # Add system prompt as separate text if present
    system_prompt = doc.get("system_prompt", "")
    if system_prompt:
        messages[0]["content"].append({"type": "text", "text": system_prompt})

    # Split by both <video> and <image> placeholders
    pattern = r"(<video>|<image>)"
    parts = re.split(pattern, prompt)

    video_idx = 0
    image_idx = 0

    for part in parts:
        if part == "<video>":
            if video_idx < len(video_frame_lists):
                # Add all frames for this video
                for frame in video_frame_lists[video_idx]:
                    messages[0]["content"].append({"type": "image", "url": frame})
                video_idx += 1
        elif part == "<image>":
            if image_idx < len(ref_image_pils):
                messages[0]["content"].append({"type": "image", "url": ref_image_pils[image_idx]})
                image_idx += 1
        elif part.strip():
            messages[0]["content"].append({"type": "text", "text": part})

    return messages


##################
# Result processing and aggregation
##################

# Mapping from dataset category names to metric keys (must match YAML metric names)
CATEGORY_TO_METRIC = {
    "(Cross-Video) Memory Update": "cross_video_memory_update",
    "(Cross-Video) Memoery Update": "cross_video_memory_update",  # Handle typo in dataset
    "(Cross-Video) Multi-View Integration": "cross_video_multi_view",
    "(Motion Understanding) Camera Motion": "motion_camera",
    "(Motion Understanding) Instance Motion": "motion_instance",
    "(Motion Understanding) Interactive Motion": "motion_interactive",
    "(Spatial Construction) Camera-Instance Spatial Relationship": "spatial_camera_instance",
    "(Spatial Construction) Camera-Scene Spatial Relationship": "spatial_camera_scene",
    "(Spatial Construction) Instance-Instance Spatial Relationship": "spatial_instance_instance",
    "(Spatial Construction) Instance-Scene Spatial Relationship": "spatial_instance_scene",
    "(Spatial Construction) Instance/Scene Attribute": "spatial_attribute",
    "(Spatial Construction) Scene-Scene Spatial Relationship": "spatial_scene_scene",
    "Planning": "planning",
    "Prediction": "prediction",
}

# All metric keys (must match YAML metric_list)
METRIC_KEYS = list(set(CATEGORY_TO_METRIC.values()))


def process_results(doc, results):
    """
    Process model results and extract accuracy metrics.
    Returns dict with separate metrics for overall and each category.
    """

    response = results[0].strip() if results else ""
    pred = extract_answer(response)
    gt = doc.get("ground_truth", "")

    # Handle ground_truth that might be stored differently
    if isinstance(gt, str):
        gt = gt.strip().upper()
        if len(gt) > 1:
            gt = gt[0]  # Take first character if it's like "A. answer"

    score = 1.0 if pred == gt else 0.0

    # Get question type/category from dataset
    question_type = doc.get("type", doc.get("question_type", ""))

    # Map to metric key
    metric_key = CATEGORY_TO_METRIC.get(question_type, None)

    if metric_key is None:
        eval_logger.warning(f"[process_results] Unknown category: {question_type}")

    # Return separate metrics for each category
    metrics = {
        # Overall accuracy - always count this sample
        "overall_accuracy": {"score": score, "total": 1},
    }

    # Add category-specific metric (only count if this sample belongs to that category)
    for key in METRIC_KEYS:
        if metric_key == key:
            metrics[key] = {"score": score, "total": 1}
        else:
            metrics[key] = {"score": 0, "total": 0}  # Not applicable for this sample

    return metrics


def _aggregate_category(results):
    """Aggregate results for a category. Results are already filtered by lmms-eval."""
    total_correct = 0
    total_count = 0

    for result in results:
        total_correct += result.get("score", 0)
        total_count += result.get("total", 0)

    if total_count == 0:
        return 0.0

    accuracy = (total_correct / total_count) * 100
    return round(accuracy, 2)


def aggregate_overall_accuracy(results):
    """Aggregate overall accuracy across all samples."""
    return _aggregate_category(results)


def aggregate_cross_video_memory_update(results):
    """Aggregate accuracy for (Cross-Video) Memory Update."""
    return _aggregate_category(results)


def aggregate_cross_video_multi_view(results):
    """Aggregate accuracy for (Cross-Video) Multi-View Integration."""
    return _aggregate_category(results)


def aggregate_motion_camera(results):
    """Aggregate accuracy for (Motion Understanding) Camera Motion."""
    return _aggregate_category(results)


def aggregate_motion_instance(results):
    """Aggregate accuracy for (Motion Understanding) Instance Motion."""
    return _aggregate_category(results)


def aggregate_motion_interactive(results):
    """Aggregate accuracy for (Motion Understanding) Interactive Motion."""
    return _aggregate_category(results)


def aggregate_spatial_camera_instance(results):
    """Aggregate accuracy for (Spatial Construction) Camera-Instance Spatial Relationship."""
    return _aggregate_category(results)


def aggregate_spatial_camera_scene(results):
    """Aggregate accuracy for (Spatial Construction) Camera-Scene Spatial Relationship."""
    return _aggregate_category(results)


def aggregate_spatial_instance_instance(results):
    """Aggregate accuracy for (Spatial Construction) Instance-Instance Spatial Relationship."""
    return _aggregate_category(results)


def aggregate_spatial_instance_scene(results):
    """Aggregate accuracy for (Spatial Construction) Instance-Scene Spatial Relationship."""
    return _aggregate_category(results)


def aggregate_spatial_attribute(results):
    """Aggregate accuracy for (Spatial Construction) Instance/Scene Attribute."""
    return _aggregate_category(results)


def aggregate_spatial_scene_scene(results):
    """Aggregate accuracy for (Spatial Construction) Scene-Scene Spatial Relationship."""
    return _aggregate_category(results)


def aggregate_planning(results):
    """Aggregate accuracy for Planning."""
    return _aggregate_category(results)


def aggregate_prediction(results):
    """Aggregate accuracy for Prediction."""
    return _aggregate_category(results)
