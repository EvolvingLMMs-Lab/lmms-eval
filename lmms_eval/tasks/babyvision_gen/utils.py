"""
BabyVision Gen Utils
Image generation evaluation task using LLM API
"""

import base64
import json
import os
from collections import defaultdict
from io import BytesIO
from typing import Dict, List, Optional

import numpy as np
from loguru import logger as eval_logger
from openai import OpenAI
from PIL import Image

from lmms_eval.tasks.babyvision_gen.prompt import build_evaluation_prompt

BABYVISION_API_KEY = os.getenv("BABYVISION_API_KEY")
BABYVISION_BASE_URL = os.getenv("BABYVISION_BASE_URL")
BABYVISION_MODEL_NAME = os.getenv("BABYVISION_MODEL_NAME", "gpt-4o")


def _get_openai_client() -> Optional[OpenAI]:
    """Get OpenAI client instance."""
    if not BABYVISION_API_KEY:
        eval_logger.error("API key not found. Set BABYVISION_API_KEY environment variable.")
        return None

    client_kwargs = {"api_key": BABYVISION_API_KEY}
    if BABYVISION_BASE_URL:
        client_kwargs["base_url"] = BABYVISION_BASE_URL

    return OpenAI(**client_kwargs)


def image_to_base64(image) -> Optional[str]:
    """Convert PIL Image or image path to base64 string"""
    try:
        if isinstance(image, str):
            if not os.path.exists(image):
                eval_logger.warning(f"Image file not found: {image}")
                return None
            with open(image, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        elif hasattr(image, "save"):
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        else:
            eval_logger.warning(f"Unknown image type: {type(image)}")
            return None
    except Exception as e:
        eval_logger.error(f"Error converting image to base64: {e}")
        return None


def parse_bool_response(response_text: str) -> Optional[bool]:
    """Parse boolean response from LLM."""
    text = response_text.strip().lower()
    if "true" in text:
        return True
    elif "false" in text:
        return False
    else:
        eval_logger.warning(f"Could not parse boolean from response: {response_text[:200]}...")
        return None


def _call_openai_for_evaluation(
    client: OpenAI,
    input_image,
    gt_image,
    generated_image,
    task_type: str,
    subtype: str,
    generation_prompt: str,
) -> Optional[str]:
    """Call OpenAI API for image generation evaluation."""
    input_b64 = image_to_base64(input_image)
    gt_b64 = image_to_base64(gt_image)
    gen_b64 = image_to_base64(generated_image)

    if not input_b64 or not gt_b64 or not gen_b64:
        eval_logger.error("Failed to convert images to base64")
        return None

    prompt = build_evaluation_prompt(task_type, subtype, generation_prompt)

    try:
        content = [
            {"type": "text", "text": "**Image 1 (Input):**"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{input_b64}"}},
            {"type": "text", "text": "\n**Image 2 (Ground Truth):**"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{gt_b64}"}},
            {"type": "text", "text": "\n**Image 3 (Generated):**"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{gen_b64}"}},
            {"type": "text", "text": "\n" + prompt},
        ]

        response = client.chat.completions.create(
            model=BABYVISION_MODEL_NAME,
            messages=[{"role": "user", "content": content}],
            max_tokens=200,
            temperature=0.0,
        )

        return response.choices[0].message.content
    except Exception as e:
        eval_logger.error(f"Error calling OpenAI API: {e}")
        return None


def babyvision_doc_to_visual(doc):
    """Extract input image from document"""
    image = doc["image"]
    return [image.convert("RGB")]


def babyvision_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Extract generation prompt text from document"""
    instruction = doc["generationPrompt"].strip()
    pre_prompt = ""
    post_prompt = ""
    if lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    return f"{pre_prompt}{instruction}{post_prompt}"


def babyvision_doc_to_target(doc):
    """Extract target instruction (for reference)"""
    return doc["generationPrompt"]


def babyvision_process_results(doc, results, **kwargs):
    """Process model predictions and evaluate using LLM API."""
    task_id = doc["taskId"]
    task_type = doc["type"]
    subtype = doc["subtype"]
    generation_prompt = doc["generationPrompt"]

    client = _get_openai_client()
    if client is None:
        return {"babyvision_overall_accuracy": {"taskId": task_id, "correct": False, "error": "No API client"}}

    pred = results[0] if results else "{}"
    try:
        pred = json.loads(pred)
    except (json.JSONDecodeError, TypeError):
        eval_logger.warning(f"Failed to parse prediction JSON: {pred}")
        return {"babyvision_overall_accuracy": {"taskId": task_id, "correct": False, "error": "Invalid JSON"}}

    model_images = pred.get("images", [])
    if not model_images:
        return {"babyvision_overall_accuracy": {"taskId": task_id, "correct": False, "error": "No generated image"}}

    input_image_pil = doc["image"].convert("RGB")
    gt_image_pil = doc["answerImage"].convert("RGB")
    generated_image_pil = Image.open(model_images[0]).convert("RGB")

    model_response = _call_openai_for_evaluation(client, input_image_pil, gt_image_pil, generated_image_pil, task_type, subtype, generation_prompt)

    if model_response is None:
        eval_logger.warning(f"Model evaluation failed for taskId {task_id}")
        return {"babyvision_overall_accuracy": {"taskId": task_id, "correct": False, "error": "Evaluation failed"}}

    correct = parse_bool_response(model_response)
    if correct is None:
        return {"babyvision_overall_accuracy": {"taskId": task_id, "correct": False, "error": "Could not parse response"}}

    return {
        "babyvision_overall_accuracy": {
            "taskId": task_id,
            "task_type": task_type,
            "subtype": subtype,
            "correct": correct,
        }
    }


def babyvision_aggregate_results(results):
    """Aggregate results across all samples."""
    if not results:
        return 0.0

    correct_count = sum(1 for r in results if r["correct"])
    total_count = len(results)

    accuracy = correct_count / total_count if total_count > 0 else 0.0

    task_type_counts = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        task_type = r["task_type"]
        subtype = r["subtype"]
        key = f"{task_type}/{subtype}"
        task_type_counts[key]["total"] += 1
        if r["correct"]:
            task_type_counts[key]["correct"] += 1

    eval_logger.info("=" * 60)
    eval_logger.info("BabyVision Gen Final Results")
    eval_logger.info("=" * 60)
    eval_logger.info(f"Overall Accuracy: {accuracy:.3f}")
    eval_logger.info(f"Total Samples: {total_count}")
    eval_logger.info(f"Correct: {correct_count}")
    eval_logger.info("-" * 60)
    eval_logger.info("Accuracy by Task Type/Subtype:")
    for key in sorted(task_type_counts.keys()):
        stats = task_type_counts[key]
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        eval_logger.info(f"  {key}: {acc:.3f} ({stats['correct']}/{stats['total']})")
    eval_logger.info("=" * 60)

    return float(accuracy)
