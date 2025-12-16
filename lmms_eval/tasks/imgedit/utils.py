"""
ImgEdit Benchmark Utils
Image editing evaluation task using OpenAI API

Based on: https://github.com/sysuyy/ImgEdit
Paper: ImgEdit: A Unified Image Editing Benchmark

Environment variables:
    - IMGEDIT_API_KEY: OpenAI API key for evaluation
    - IMGEDIT_BASE_URL: Optional custom OpenAI API base URL
    - IMGEDIT_MODEL_NAME: Model name for evaluation
"""

import base64
import json
import os
import re
from collections import defaultdict
from io import BytesIO
from typing import Dict, Optional, Tuple

import numpy as np
from loguru import logger as eval_logger
from openai import OpenAI
from PIL import Image

from lmms_eval.tasks.imgedit.prompt import IMGEDIT_PROMPTS

# Environment variable names for OpenAI API configuration
IMGEDIT_API_KEY = os.getenv("IMGEDIT_API_KEY")
IMGEDIT_BASE_URL = os.getenv("IMGEDIT_BASE_URL")
IMGEDIT_MODEL_NAME = os.getenv("IMGEDIT_MODEL_NAME", "gpt-4o")

# Global OpenAI client (singleton)
_openai_client = None


def _get_openai_client() -> Optional[OpenAI]:
    """Get or create OpenAI client instance (singleton pattern)."""
    global _openai_client
    if _openai_client is not None:
        return _openai_client

    if not IMGEDIT_API_KEY:
        eval_logger.error("API key not found. Set IMGEDIT_API_KEY environment variable.")
        return None

    client_kwargs = {"api_key": IMGEDIT_API_KEY}
    if IMGEDIT_BASE_URL:
        client_kwargs["base_url"] = IMGEDIT_BASE_URL

    _openai_client = OpenAI(**client_kwargs)
    eval_logger.info(f"Initialized OpenAI client (model: {IMGEDIT_MODEL_NAME})")
    return _openai_client


IMGEDIT_EDIT_TYPES = [
    "replace",
    "add",
    "adjust",
    "remove",
    "style",
    "action",
    "extract",
    "background",
    "compose",
]


def image_to_base64(image) -> Optional[str]:
    """Convert PIL Image or image path to base64 string"""
    try:
        if isinstance(image, str):
            # It's a path
            if not os.path.exists(image):
                eval_logger.warning(f"Image file not found: {image}")
                return None
            with open(image, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        elif hasattr(image, "save"):
            # It's a PIL Image
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        else:
            eval_logger.warning(f"Unknown image type: {type(image)}")
            return None
    except Exception as e:
        eval_logger.error(f"Error converting image to base64: {e}")
        return None


def parse_gpt_scores(response_text: str) -> Tuple[float, float, float]:
    """
    Parse GPT/Qwen response to extract three scores.
    Returns tuple of (score1, score2, score3)
    """
    try:
        # Find all numbers in the format "Score Name: X"
        score_pattern = r":\s*(\d+)"
        matches = re.findall(score_pattern, response_text)

        if len(matches) >= 3:
            # Take the last 3 numbers (the actual scores)
            scores = [float(matches[-3]), float(matches[-2]), float(matches[-1])]
            return tuple(scores)

        # Alternative: find standalone numbers on lines
        lines = response_text.strip().split("\n")
        scores = []
        for line in lines:
            # Look for patterns like "Prompt Compliance: 4" or just "4"
            match = re.search(r"(\d+)\s*$", line.strip())
            if match:
                scores.append(float(match.group(1)))

        if len(scores) >= 3:
            return (scores[-3], scores[-2], scores[-1])

        eval_logger.warning(f"Could not parse 3 scores from response: {response_text[:200]}...")
        return (0.0, 0.0, 0.0)
    except Exception as e:
        eval_logger.error(f"Error parsing scores: {e}")
        return (0.0, 0.0, 0.0)


def calculate_average_score(scores: Tuple[float, float, float]) -> float:
    """Calculate average of three scores"""
    return sum(scores) / 3.0


def imgedit_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def imgedit_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Extract instruction text from document"""
    instruction = doc.get("prompt", "").strip()
    pre_prompt = ""
    post_prompt = ""
    if lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    return f"{pre_prompt}{instruction}{post_prompt}"


def imgedit_doc_to_target(doc):
    """Extract target instruction (for reference)"""
    return doc.get("prompt", "")


# ============================================
# OpenAI Evaluation Backend
# ============================================


def _call_openai_for_evaluation(
    client: OpenAI,
    original_image,
    edited_image,
    edit_prompt: str,
    edit_type: str,
) -> Optional[str]:
    """
    Call OpenAI API for image editing evaluation.

    Args:
        client: OpenAI client instance
        original_image: Original image (PIL Image or path)
        edited_image: Edited image (PIL Image or path)
        edit_prompt: The editing instruction
        edit_type: Type of edit (replace, add, adjust, etc.)

    Returns:
        Model response text or None if failed
    """
    # Convert images to base64
    original_b64 = image_to_base64(original_image)
    edited_b64 = image_to_base64(edited_image)

    if not original_b64 or not edited_b64:
        eval_logger.error("Failed to convert images to base64")
        return None

    # Get prompt template for this edit type
    prompt_template = IMGEDIT_PROMPTS.get(edit_type, IMGEDIT_PROMPTS["adjust"])
    full_prompt = prompt_template.replace("<edit_prompt>", edit_prompt)

    try:
        # Build message content
        content = [
            {"type": "text", "text": full_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{original_b64}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{edited_b64}"}},
        ]

        response = client.chat.completions.create(
            model=IMGEDIT_MODEL_NAME,
            messages=[{"role": "user", "content": content}],
            max_tokens=512,
        )

        return response.choices[0].message.content
    except Exception as e:
        eval_logger.error(f"Error calling OpenAI API: {e}")
        return None


# ============================================
# Process Results
# ============================================


def imgedit_process_results(doc, results, **kwargs):
    """
    Process model predictions and evaluate using OpenAI API.

    Args:
        doc: Document containing image, prompt, key, edit_type
        results: Model predictions [JSON string with {"text": "...", "images": [...]}]
        **kwargs: Additional arguments

    Returns:
        Dict with metrics: imgedit_score1, imgedit_score2, imgedit_score3, imgedit_avg_score
    """
    # Extract document fields
    key = doc.get("key", "unknown")
    edit_type = doc.get("edit_type", "adjust")
    edit_prompt = doc.get("prompt", "")

    # Get OpenAI client (singleton, initialized once)
    client = _get_openai_client()
    import pdb

    pdb.set_trace()
    if client is None:
        return _create_zero_result(key, edit_type)

    # Parse prediction JSON
    pred = results[0] if results else "{}"
    try:
        pred = json.loads(pred)
    except (json.JSONDecodeError, TypeError):
        eval_logger.warning(f"Failed to parse prediction JSON: {pred}")
        return _create_zero_result(key, edit_type)

    model_images = pred.get("images", [])

    # Get input image from doc
    input_image_pil = doc.get("image").convert("RGB")
    edited_image_pil = Image.open(model_images[0]).convert("RGB")

    # Call OpenAI for evaluation
    model_response = _call_openai_for_evaluation(
        client,
        input_image_pil,
        edited_image_pil,
        edit_prompt,
        edit_type,
    )

    if model_response is None:
        eval_logger.warning(f"Model evaluation failed for key {key}")
        return _create_zero_result(key, edit_type)

    # Parse scores from model response
    score1, score2, score3 = parse_gpt_scores(model_response)
    avg_score = calculate_average_score((score1, score2, score3))

    return {
        "imgedit_score1": {
            "key": key,
            "edit_type": edit_type,
            "score": float(score1),
        },
        "imgedit_score2": {
            "key": key,
            "edit_type": edit_type,
            "score": float(score2),
        },
        "imgedit_score3": {
            "key": key,
            "edit_type": edit_type,
            "score": float(score3),
        },
        "imgedit_avg_score": {
            "key": key,
            "edit_type": edit_type,
            "score": float(avg_score),
            "model_response": model_response,
        },
    }


def _create_zero_result(key: str, edit_type: str) -> Dict:
    """Create a zero-score result dict"""
    return {
        "imgedit_score1": {
            "key": key,
            "edit_type": edit_type,
            "score": 0.0,
        },
        "imgedit_score2": {
            "key": key,
            "edit_type": edit_type,
            "score": 0.0,
        },
        "imgedit_score3": {
            "key": key,
            "edit_type": edit_type,
            "score": 0.0,
        },
        "imgedit_avg_score": {
            "key": key,
            "edit_type": edit_type,
            "score": 0.0,
        },
    }


# ============================================
# Aggregation Functions
# ============================================


def imgedit_aggregate_score(results):
    """
    Simple aggregation: compute mean of all scores.
    Used for imgedit_score1, imgedit_score2, imgedit_score3.
    """
    if not results:
        return 0.0
    scores = [r["score"] for r in results if "score" in r]
    if not scores:
        return 0.0
    return float(np.mean(scores))


def imgedit_aggregate_avg_score(results):
    """
    Aggregate avg_score (the main metric) and print final summary.
    This follows the original ImgEdit benchmark logic:
    - Step 1: Each sample has avg_score = (score1 + score2 + score3) / 3
    - Step 2: Group by edit_type and compute type averages
    - Final: Overall average of all avg_scores
    """
    if not results:
        return 0.0

    scores = [r["score"] for r in results if "score" in r]
    if not scores:
        return 0.0

    overall_avg = float(np.mean(scores))

    # Compute per-type averages (like step2_typescore.py)
    edit_type_scores = defaultdict(list)
    for r in results:
        if "score" in r:
            edit_type = r.get("edit_type", "unknown")
            edit_type_scores[edit_type].append(r["score"])

    # Log final summary in original ImgEdit format
    eval_logger.info("=" * 60)
    eval_logger.info("ImgEdit Benchmark Final Results")
    eval_logger.info("=" * 60)
    eval_logger.info(f"Overall Average Score: {overall_avg:.3f}")
    eval_logger.info(f"Total Samples: {len(scores)}")
    eval_logger.info("-" * 60)
    eval_logger.info("Scores by Edit Type (avg_score):")
    for edit_type in IMGEDIT_EDIT_TYPES:
        if edit_type in edit_type_scores:
            type_avg = np.mean(edit_type_scores[edit_type])
            eval_logger.info(f"  {edit_type:12s}: {type_avg:.3f} (n={len(edit_type_scores[edit_type])})")
    eval_logger.info("=" * 60)

    return overall_avg
