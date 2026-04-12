"""
UniG2U Benchmark - Merged utilities for all sub-tasks.
"""

import base64
import io
import json
import logging
import os
import random
import re
import time
from collections import defaultdict
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
from loguru import logger as eval_logger
from PIL import Image

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from openai import AzureOpenAI, OpenAI
except ImportError:
    AzureOpenAI = None
    OpenAI = None
    eval_logger.debug("openai not installed, LLM-judge tasks will not work")

AZURE_AVAILABLE = False
try:
    from azure.identity import (
        AzureCliCredential,
        ChainedTokenCredential,
        ManagedIdentityCredential,
        get_bearer_token_provider,
    )

    AZURE_AVAILABLE = True
except ImportError:
    eval_logger.debug("azure-identity not installed")

try:
    from lmms_eval.azure_openai_compat import build_client as build_azure_compat_client
    from lmms_eval.azure_openai_compat import has_endpoint_support
except ImportError:
    eval_logger.debug("azure_openai_compat not available")

    def has_endpoint_support():
        return False

    def build_azure_compat_client(*args, **kwargs):
        raise ImportError("azure_openai_compat not installed")

try:
    from lmms_eval.filters.extraction import ExtendedRegexFilter
    from lmms_eval.filters.transformation import MapFilter
except ImportError:
    eval_logger.debug("lmms_eval filters not available")


# ============================================================================
# Shared LLM Judge Client
# ============================================================================

def _get_azure_cli_token(resource: str = "https://cognitiveservices.azure.com/") -> str:
    """Get Azure bearer token via `az account get-access-token`."""
    import subprocess
    _token_cache = getattr(_get_azure_cli_token, "_cache", {"token": None, "expires_at": 0})
    now = time.time()
    if _token_cache["token"] and now < _token_cache["expires_at"] - 300:
        return _token_cache["token"]
    payload = json.loads(subprocess.check_output(
        ["az", "account", "get-access-token", "--resource", resource, "-o", "json"],
        text=True,
    ))
    _token_cache["token"] = payload["accessToken"]
    _token_cache["expires_at"] = float(payload.get("expires_on") or payload.get("expiresOn") or 0)
    _get_azure_cli_token._cache = _token_cache
    return _token_cache["token"]


class _JudgeClient:
    """Unified LLM Judge client. Supports Azure CLI endpoint, TRAPI, or OpenAI."""

    def __init__(self) -> None:
        # Priority: AZURE_OPENAI_ENDPOINT > OPENAI_API_KEY > azure_compat > TRAPI
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        openai_key = os.getenv("OPENAI_API_KEY")

        if azure_endpoint:
            # Azure OpenAI with CLI token auth
            token = _get_azure_cli_token()
            self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
            self.client = AzureOpenAI(
                azure_endpoint=azure_endpoint.replace("/openai/v1", ""),
                api_key=token,
                api_version=api_version,
            )
            eval_logger.info(f"LLM Judge: Azure CLI endpoint ({azure_endpoint})")
        elif openai_key:
            base_url = os.getenv("OPENAI_BASE_URL")
            self.deployment = os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o")
            self.client = OpenAI(api_key=openai_key, **({"base_url": base_url} if base_url else {}))
            eval_logger.info("LLM Judge: OpenAI API")
        elif AZURE_AVAILABLE and has_endpoint_support():
            client, self.deployment = build_azure_compat_client()
            self.client = client
            eval_logger.info("LLM Judge: Azure compat endpoint")
        elif AZURE_AVAILABLE:
            scope = os.getenv("TRAPI_SCOPE", "api://trapi/.default")
            self.deployment = os.getenv("TRAPI_DEPLOYMENT", "gpt-4o_2024-11-20")
            instance = os.getenv("TRAPI_INSTANCE", "gcr/shared")
            api_version = os.getenv("TRAPI_API_VERSION", "2024-10-21")
            endpoint = f"https://trapi.research.microsoft.com/{instance}"
            credential_provider = get_bearer_token_provider(
                ChainedTokenCredential(AzureCliCredential(), ManagedIdentityCredential()),
                scope,
            )
            self.client = AzureOpenAI(
                azure_endpoint=endpoint,
                azure_ad_token_provider=credential_provider,
                api_version=api_version,
            )
            eval_logger.info("LLM Judge: Azure TRAPI")
        else:
            raise RuntimeError(
                "No LLM judge backend available. Set one of:\n"
                "  AZURE_OPENAI_ENDPOINT (Azure CLI auth)\n"
                "  OPENAI_API_KEY (OpenAI)\n"
                "  or install azure-identity for TRAPI"
            )

    def chat_completion(self, *, messages, **kwargs) -> str:
        resp = self.client.chat.completions.create(
            model=self.deployment, messages=messages, **kwargs
        )
        return resp.choices[0].message.content


_JUDGE_CLIENT: Optional[_JudgeClient] = None


def _get_judge_client() -> _JudgeClient:
    global _JUDGE_CLIENT
    if _JUDGE_CLIENT is None:
        _JUDGE_CLIENT = _JudgeClient()
    return _JUDGE_CLIENT


# Alias for sub-tasks that use get_judge_client (without underscore prefix)
get_judge_client = _get_judge_client



# ======================================================================
# chartqa
# ======================================================================

def chartqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def chartqa_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"


def chartqa_process_results(doc, results):
    pred = results[0]
    type = doc["type"]
    score = relaxed_correctness(pred, doc["answer"])
    score = 1.0 if score else 0.0
    return_dict = {"relaxed_overall": score}
    if type == "human_test":
        return_dict["relaxed_human_split"] = score
    else:
        return_dict["relaxed_augmented_split"] = score
    return return_dict


def relaxed_correctness(prediction, target, max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    "Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct."

    This funcion is taken from https://github.com/QwenLM/Qwen-VL/blob/34b4c0ee7b07726371b960911f249fe61b362ca3/eval_mm/evaluate_vqa.py#L113
    Args:
      target: List of target string.
      prediction: List of predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str):
        try:
            if text.endswith("%"):
                # Convert percentages to floats.
                return float(text.rstrip("%")) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()


# ============================================================================
# Visual CoT Version
# ============================================================================

CHARTQA_GEN_PROMPT = (
    "Based on this chart and the question being asked, generate an annotated version "
    "of the chart that highlights the relevant data points, bars, lines, or regions "
    "needed to answer the question. Mark or circle the key values and label them clearly."
)


def chartqa_doc_to_text_visual_cot(doc, lmms_eval_specific_kwargs):
    """Visual CoT prompt for ChartQA task."""
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    question_with_aux = (
        "In addition to the original chart, you are also given an annotated version "
        "that highlights the relevant data points for the question.\n\n"
        f"{pre_prompt}{question}{post_prompt}"
    )

    return f"[GEN_PROMPT]{CHARTQA_GEN_PROMPT}[/GEN_PROMPT][QUESTION]{question_with_aux}[/QUESTION]"


# ======================================================================
# mmsi_bench
# ======================================================================



def msr_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"].strip()
    if "pre_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["pre_prompt"] != "":
        question = f"{lmms_eval_specific_kwargs['pre_prompt']}{question}"
    if "post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["post_prompt"] != "":
        question = f"{question}{lmms_eval_specific_kwargs['post_prompt']}"
    return question


def msr_doc_to_text_with_gen_prompt(doc, lmms_eval_specific_kwargs=None):
    """
    Version for visual CoT models that includes generation prompt in a parseable format.

    Output format: [GEN_PROMPT]...[/GEN_PROMPT][QUESTION]...[/QUESTION]
    - Stage 1 uses content inside [GEN_PROMPT] tags
    - Stage 2 uses content inside [QUESTION] tags (includes pre_prompt and post_prompt)
    """
    question = doc["question"].strip()

    # Build Stage 2 question with pre_prompt and post_prompt
    stage2_question = question
    if "pre_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["pre_prompt"] != "":
        stage2_question = f"{lmms_eval_specific_kwargs['pre_prompt']}{stage2_question}"
    if "post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["post_prompt"] != "":
        stage2_question = f"{stage2_question}{lmms_eval_specific_kwargs['post_prompt']}"

    # Add generation prompt as a special marker if provided
    if "generation_prompt" in lmms_eval_specific_kwargs:
        gen_prompt = lmms_eval_specific_kwargs["generation_prompt"]
        # Format: [GEN_PROMPT]...[/GEN_PROMPT][QUESTION]...[/QUESTION]
        return f"[GEN_PROMPT]{gen_prompt}[/GEN_PROMPT][QUESTION]{stage2_question}[/QUESTION]"

    return stage2_question


def msr_doc_to_visual(doc):
    image_list = []
    for img_data in doc["images"]:
        # Check if already a PIL Image object
        if isinstance(img_data, Image.Image):
            image = img_data.convert("RGB")
        elif isinstance(img_data, dict):
            # If dict (from HuggingFace datasets), extract bytes
            image = Image.open(io.BytesIO(img_data["bytes"]))
            image = image.convert("RGB")
        else:
            # If bytes data, decode it
            image = Image.open(io.BytesIO(img_data))
            image = image.convert("RGB")
        image_list.append(image)
    return image_list


def extract_single_choice_with_word_boundary(pred, gt):
    pattern_1 = r"``([^`]*)``"
    match = re.search(pattern_1, pred)
    if match:
        pred = match.group(1)

    pattern_2 = r"`([^`]*)`"
    match = re.search(pattern_2, pred)
    if match:
        pred = match.group(1)

    pattern_add = r"\{([^}]*)\}"
    match = re.search(pattern_add, pred)
    if match:
        pred = match.group(1)

    pattern_3 = r"\b[A-D]\b(?!\s[a-zA-Z])"
    match = re.search(pattern_3, pred)
    if match:
        pred = match.group()
    else:
        return None

    answer = gt.lower().replace("\n", " ").strip()
    predict = pred.lower().replace("\n", " ").strip()
    try:
        if answer == predict[0]:
            return 1.0
        elif predict[0] == "(" and answer == predict[1]:
            return 1.0
        elif predict[0:7] == "option " and answer == predict[7]:
            return 1.0
        elif predict[0:14] == "the answer is " and answer == predict[14]:
            return 1.0
    except Exception as e:
        return 0.0
    return 0.0


def msr_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name, value: metric value
    """
    pred = results[0]
    gt = doc["answer"]

    score = extract_single_choice_with_word_boundary(pred, gt)
    category = doc["question_type"]
    l2_category = doc["question_type"]
    if score is None:
        return {category: {"question_id": doc["id"], "l2_category": l2_category, "score": 0, "note": "can not find anwser"}, "average": {"question_id": doc["id"], "l2_category": l2_category, "score": 0, "note": "can not find anwser"}}
    return {category: {"question_id": doc["id"], "l2_category": l2_category, "score": score}, "average": {"question_id": doc["id"], "l2_category": l2_category, "score": score}}


def msr_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    l2_category_scores = defaultdict(list)
    for result in results:
        score = result["score"]
        l2_category = result["l2_category"]
        l2_category_scores[l2_category].append(score)

    l2_category_avg_score = {}
    for l2_category, scores in l2_category_scores.items():
        avg_score = sum(scores) / len(scores)
        l2_category_avg_score[l2_category] = avg_score
        eval_logger.info(f"{l2_category}: {avg_score:.2f}")

    all_scores = [score for scores in l2_category_scores.values() for score in scores]
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    return avg_score


# ======================================================================
# vsp
# ======================================================================

"""
Visual-Spatial-Planning (VSP) Task Utilities
Tasks: Google Map Navigation, Collision Detection
"""




# ============================================================================
# Google Map Task
# ============================================================================

GMAP_PROMPT = '''As a professional pathfinder, your task is to analyze a map and find a route from the starting location to the goal. Since coding is not within your skill set, your approach relies on logical reasoning of the map.

## Game Setup
- The game presents a fully observable map.
- The starting location is marked with blue "S", and the goal is marked with red "G".
- Your goal is to find a path from the starting location to the goal.

## Moving Rules
- The action plan involves moves in four directions: 'W' (West), 'E' (east), 'N' (north), or 'S' (south).
- Each move is along with distances. Distances are measured by how many crossroads passed.
We provide an example to further illustrate the rules.

[Example Image]

In this provided example:
- You are now at the southwest of the goal.
- If you move north by 1 crossroad, you will be at the west of the goal.
- If you move east by 4 crossroads, you will be at the goal.
- IMPORTANT: Please ignore the name of the street and avenue. The numbers in the name cannot be used to compute how many crossroads need to be passed.

## Procedure and Output
Now you will solve the given maze. To analyze the relative spatial relation between the starting point and the goal (for example, southwest). Then, output a path using the format <Direction>: <Number of crossroads passed>.
For example:
<Output>
1. North: 1
2. East: 4
means move north by 1 crossroad, and move east by 4 crossroads.
<Output>
1. South: 1
means move south by 1 crossroad.
Do not output any extra content after the above aggregated output.

Please output path for the following map:

[Test Image]'''


def gmap_doc_to_visual(doc: Dict) -> List[Image.Image]:
    """Get visual inputs for google map task."""
    images = []
    for key in ["example_image", "test_image"]:
        if key in doc and doc[key]:
            img_data = doc[key]
            if isinstance(img_data, bytes):
                images.append(Image.open(BytesIO(img_data)).convert("RGB"))
            elif isinstance(img_data, Image.Image):
                images.append(img_data.convert("RGB"))
    return images


def gmap_doc_to_text(doc: Dict, lmms_eval_specific_kwargs: Dict = None) -> str:
    """Get prompt for google map task."""
    return GMAP_PROMPT


def gmap_process_results(doc: Dict, results: List[str]) -> Dict[str, Any]:
    """Process google map results - parse directions and compare."""
    result_text = results[0] if results else ""

    # Parse directions
    north_south = 0
    east_west = 0

    try:
        # Find Output section
        c_index = result_text.find("Output")
        if c_index == -1:
            c_index = result_text.find("Path")
        if c_index != -1:
            contents = result_text[c_index + 7:]
        else:
            contents = result_text

        contents = contents.replace('"', '').replace("'", '').replace(".", '')
        lines = contents.strip().split('\n')

        for line in lines:
            line_lower = line.lower()
            if "north" in line_lower:
                match = re.search(r'north[:\s]*(\d+)', line_lower)
                if match:
                    north_south += int(match.group(1))
            if "south" in line_lower:
                match = re.search(r'south[:\s]*(\d+)', line_lower)
                if match:
                    north_south -= int(match.group(1))
            if "east" in line_lower:
                match = re.search(r'east[:\s]*(\d+)', line_lower)
                if match:
                    east_west += int(match.group(1))
            if "west" in line_lower:
                match = re.search(r'west[:\s]*(\d+)', line_lower)
                if match:
                    east_west -= int(match.group(1))
    except Exception as e:
        eval_logger.error(f"Error parsing google map result: {e}")

    gt_ns = doc.get("gt_north_south", 0)
    gt_ew = doc.get("gt_east_west", 0)

    correct = 1.0 if (north_south == gt_ns and east_west == gt_ew) else 0.0

    return {
        "gmap_acc": {
            "test_id": doc.get("test_id"),
            "score": correct,
            "pred_ns": north_south,
            "pred_ew": east_west,
            "gt_ns": gt_ns,
            "gt_ew": gt_ew,
        }
    }


def gmap_aggregate_results(results: List[Dict]) -> float:
    """Aggregate google map results."""
    scores = [r["score"] for r in results]
    acc = sum(scores) / len(scores) if scores else 0.0
    eval_logger.info(f"Google Map Accuracy: {acc:.4f} ({sum(scores):.0f}/{len(scores)})")
    return acc


# ============================================================================
# Collision Task
# ============================================================================

COLLISION_PROMPT_TEMPLATE = '''As a professional navigation agent, your task is to analyze a map and determine the time needed for the car and the person passing the goal.

## Game Setup
- The game presents a fully observable map. There is a person, a car, and a goal on the map.
- The game further specifies the moving direction of the person and car ("up", "down", "left", "right").
- Your goal is to determine the time needed for the car and the person passing the goal.
The following figure shows how the player, the car, and the goals look like.

[Icon Image]

We provide an example to further illustrate the rules.

[Example Image]

The car is moving left with speed 1.0 grid per second, and the person is moving up with speed 0.5 grid per second.

In this provided example:
- The car is 2 grid away from the goal. Given it's time as 1.0 grid per second, the time needed is 2 / 1.0 = 2 seconds.
- The person is 1 grid away from the goal. Given it's time as 0.5 grid per second, the time needed is 1 / 0.5 = 2 seconds.

## Procedure and Output
Now you will answer for the following given map. To solve it, analyze the car and the person separately. Then, answer for them separately. For example:
Car: 2.0
Person: 2.0
means car and the person will need 2.0 seconds to pass the goal respectively.
Do not output any extra content after the above aggregated output.

Please analyze and determine the time needed for the car and the person passing the goal:

[Test Image]

The car is moving {car_dir} with speed {car_speed}, and the person is moving {person_dir} with speed {person_speed}.'''


def collision_doc_to_visual(doc: Dict) -> List[Image.Image]:
    """Get visual inputs for collision task."""
    images = []
    for key in ["icon_image", "example_image", "test_image"]:
        if key in doc and doc[key]:
            img_data = doc[key]
            if isinstance(img_data, bytes):
                images.append(Image.open(BytesIO(img_data)).convert("RGB"))
            elif isinstance(img_data, Image.Image):
                images.append(img_data.convert("RGB"))
    return images


def collision_doc_to_text(doc: Dict, lmms_eval_specific_kwargs: Dict = None) -> str:
    """Get prompt for collision task."""
    return COLLISION_PROMPT_TEMPLATE.format(
        car_dir=doc.get("car_dir", ""),
        car_speed=doc.get("car_speed", ""),
        person_dir=doc.get("person_dir", ""),
        person_speed=doc.get("person_speed", ""),
    )


def collision_process_results(doc: Dict, results: List[str]) -> Dict[str, Any]:
    """Process collision results - parse Car/Person times and compare."""
    result_text = results[0] if results else ""

    # Parse Car and Person times
    pattern = r'(Car|Person):\s*([0-9]*\.?[0-9]+)'
    matches = re.findall(pattern, result_text, re.IGNORECASE)
    info = {match[0].capitalize(): float(match[1]) for match in matches}

    gt_car = doc.get("gt_car", 0.0)
    gt_person = doc.get("gt_person", 0.0)

    pred_car = info.get("Car", -999)
    pred_person = info.get("Person", -999)

    # Correct if within 1.0 tolerance
    car_correct = 1.0 if abs(pred_car - gt_car) <= 1.0 else 0.0
    person_correct = 1.0 if abs(pred_person - gt_person) <= 1.0 else 0.0

    return {
        "collision_acc": {
            "test_id": doc.get("test_id"),
            "car_score": car_correct,
            "person_score": person_correct,
            "score": (car_correct + person_correct) / 2,
            "pred_car": pred_car,
            "pred_person": pred_person,
            "gt_car": gt_car,
            "gt_person": gt_person,
        }
    }


def collision_aggregate_results(results: List[Dict]) -> float:
    """Aggregate collision results."""
    car_scores = [r["car_score"] for r in results]
    person_scores = [r["person_score"] for r in results]

    car_acc = sum(car_scores) / len(car_scores) if car_scores else 0.0
    person_acc = sum(person_scores) / len(person_scores) if person_scores else 0.0
    overall_acc = (car_acc + person_acc) / 2

    eval_logger.info(f"  Car Accuracy: {car_acc:.4f} ({sum(car_scores):.0f}/{len(car_scores)})")
    eval_logger.info(f"  Person Accuracy: {person_acc:.4f} ({sum(person_scores):.0f}/{len(person_scores)})")
    eval_logger.info(f"  Overall Accuracy: {overall_acc:.4f}")

    return overall_acc


# ============================================================================
# Visual CoT Versions
# ============================================================================

GMAP_GEN_PROMPT = (
    "This image shows a map with a starting point marked 'S' (blue) and a goal marked 'G' (red). "
    "Your task: Generate a clear visualization that highlights the optimal path from S to G. "
    "Draw a simplified map showing the route with arrows indicating directions (North, South, East, West) "
    "and mark the number of crossroads to pass in each direction."
)

GMAP_QUESTION_PROMPT_COT = '''In addition to the original images, you are also given an auxiliary visualization image showing the path analysis.

As a professional pathfinder, your task is to analyze a map and find a route from the starting location to the goal. Since coding is not within your skill set, your approach relies on logical reasoning of the map.

## Game Setup
- The game presents a fully observable map.
- The starting location is marked with blue "S", and the goal is marked with red "G".
- Your goal is to find a path from the starting location to the goal.

## Moving Rules
- The action plan involves moves in four directions: 'W' (West), 'E' (east), 'N' (north), or 'S' (south).
- Each move is along with distances. Distances are measured by how many crossroads passed.
We provide an example to further illustrate the rules.

[Example Image]

In this provided example:
- You are now at the southwest of the goal.
- If you move north by 1 crossroad, you will be at the west of the goal.
- If you move east by 4 crossroads, you will be at the goal.
- IMPORTANT: Please ignore the name of the street and avenue. The numbers in the name cannot be used to compute how many crossroads need to be passed.

## Procedure and Output
Now you will solve the given maze. To analyze the relative spatial relation between the starting point and the goal (for example, southwest). Then, output a path using the format <Direction>: <Number of crossroads passed>.
For example:
<Output>
1. North: 1
2. East: 4
means move north by 1 crossroad, and move east by 4 crossroads.
<Output>
1. South: 1
means move south by 1 crossroad.
Do not output any extra content after the above aggregated output.

Please output path for the following map:

[Test Image]'''


def gmap_doc_to_text_visual_cot(doc: Dict, lmms_eval_specific_kwargs: Dict = None) -> str:
    """Get Visual CoT prompt for google map task."""
    return f"[GEN_PROMPT]{GMAP_GEN_PROMPT}[/GEN_PROMPT][QUESTION]{GMAP_QUESTION_PROMPT_COT}[/QUESTION]"


COLLISION_GEN_PROMPT = (
    "This image shows a map with a car, a person, and a goal. "
    "Your task: Generate a clear visualization that analyzes the distances and paths. "
    "Draw a simplified diagram showing the grid distances from the car to the goal "
    "and from the person to the goal, with arrows indicating their movement directions."
)

COLLISION_QUESTION_TEMPLATE_COT = '''In addition to the original images, you are also given an auxiliary visualization image showing the distance analysis.

As a professional navigation agent, your task is to analyze a map and determine the time needed for the car and the person passing the goal.

## Game Setup
- The game presents a fully observable map. There is a person, a car, and a goal on the map.
- The game further specifies the moving direction of the person and car ("up", "down", "left", "right").
- Your goal is to determine the time needed for the car and the person passing the goal.
The following figure shows how the player, the car, and the goals look like.

[Icon Image]

We provide an example to further illustrate the rules.

[Example Image]

The car is moving left with speed 1.0 grid per second, and the person is moving up with speed 0.5 grid per second.

In this provided example:
- The car is 2 grid away from the goal. Given it's time as 1.0 grid per second, the time needed is 2 / 1.0 = 2 seconds.
- The person is 1 grid away from the goal. Given it's time as 0.5 grid per second, the time needed is 1 / 0.5 = 2 seconds.

## Procedure and Output
Now you will answer for the following given map. To solve it, analyze the car and the person separately. Then, answer for them separately. For example:
Car: 2.0
Person: 2.0
means car and the person will need 2.0 seconds to pass the goal respectively.
Do not output any extra content after the above aggregated output.

Please analyze and determine the time needed for the car and the person passing the goal:

[Test Image]

The car is moving {car_dir} with speed {car_speed}, and the person is moving {person_dir} with speed {person_speed}.'''


def collision_doc_to_text_visual_cot(
    doc: Dict, lmms_eval_specific_kwargs: Dict = None
) -> str:
    """Get Visual CoT prompt for collision task."""
    question = COLLISION_QUESTION_TEMPLATE_COT.format(
        car_dir=doc.get("car_dir", ""),
        car_speed=doc.get("car_speed", ""),
        person_dir=doc.get("person_dir", ""),
        person_speed=doc.get("person_speed", ""),
    )
    return f"[GEN_PROMPT]{COLLISION_GEN_PROMPT}[/GEN_PROMPT][QUESTION]{question}[/QUESTION]"


# ======================================================================
# auxsolidmath
# ======================================================================

"""
AuxSolidMath Task Utilities
Evaluation for solid geometry problems with auxiliary line construction.
"""







def _find_first_json_substring(text: str) -> Optional[str]:
    """Extract first JSON object from text"""
    if not text:
        return None
    start_index = text.find("{")
    if start_index == -1:
        return None

    brace_depth, in_string, is_escaped = 0, False, False
    for i in range(start_index, len(text)):
        char = text[i]
        if char == '"' and not is_escaped:
            in_string = not in_string
        if not in_string:
            if char == "{":
                brace_depth += 1
            elif char == "}":
                brace_depth -= 1
                if brace_depth == 0:
                    return text[start_index : i + 1]
        is_escaped = char == "\\" and not is_escaped
    return None


def auxsolidmath_doc_to_visual(doc: Dict) -> List[Image.Image]:
    """Get visual input for auxsolidmath task (original diagram)"""
    original_image = doc.get("original_image")
    if original_image is not None:
        if isinstance(original_image, Image.Image):
            return [original_image]
    return []


def auxsolidmath_doc_to_text(doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None) -> str:
    """Get text prompt for auxsolidmath task"""
    question = doc.get("question", "")
    return f"""You are given a solid geometry problem with a 3D diagram.

Problem: {question}

Instructions:
1. First, carefully analyze the 3D diagram and identify what auxiliary lines (辅助线) need to be drawn to solve this problem. Common auxiliary constructions in solid geometry include:
   - Connecting points to form line segments
   - Drawing perpendiculars from a point to a plane or line
   - Finding midpoints and connecting them
   - Extending lines to find intersections
   - Drawing parallel lines through specific points
   - Constructing cross-sections

2. Clearly state which auxiliary lines you will draw and why they are helpful. For example:
   - "Connect point A to point B to form segment AB"
   - "Draw a perpendicular from point P to plane ABC, with foot H"
   - "Take the midpoint M of edge AB, connect M to C"
   - "Extend line DE to intersect plane ABC at point F"

3. After describing the auxiliary lines, provide a step-by-step solution using these auxiliary constructions.

4. Show all intermediate calculations and reasoning, including:
   - Distance calculations
   - Angle calculations
   - Volume/area calculations if needed

5. State the final answer clearly.

Please think step by step, starting with the auxiliary line construction."""


def auxsolidmath_doc_to_target(doc: Dict) -> str:
    """Get target answer for auxsolidmath task"""
    return doc.get("answer", "")


def auxsolidmath_doc_to_text_visual_cot(
    doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None
) -> str:
    """
    Get two-stage Visual Chain-of-Thought prompt for auxsolidmath task.

    Stage 1: Analyze the problem and draw auxiliary lines on the 3D diagram
    Stage 2: Solve the problem using both original diagram and the auxiliary diagram
    """
    question = doc.get("question", "")

    # Stage 1: Analyze problem and generate diagram with auxiliary constructions
    generation_prompt = f"""You are given a solid geometry problem with a 3D diagram. Analyze the problem and create an enhanced version of the SAME diagram with auxiliary constructions added.

Problem: {question}

Instructions:
1. KEEP all original elements exactly as they are (all points, edges, faces, labels)
2. Analyze what auxiliary constructions would help solve this problem
3. ADD auxiliary lines in a different color (e.g., red or dashed lines). Common auxiliary constructions include:
   - Perpendiculars from a point to a plane or line
   - Line segments connecting specific points
   - Midpoints of edges with connections
   - Extended lines to find intersections
   - Parallel lines through specific points
   - Cross-sections of the solid
4. Label any new points you add (use letters not already in the diagram)
5. The final diagram should look like the original 3D figure with extra auxiliary lines drawn on top

Generate an enhanced 3D diagram that preserves the original and adds helpful auxiliary constructions."""

    # Stage 2: Solve using both original and auxiliary diagram
    question_prompt = f"""You are given a solid geometry problem.

Problem: {question}

You are given TWO images:
1) ORIGINAL DIAGRAM: The 3D solid geometry figure as given
2) AUXILIARY DIAGRAM: The same figure with auxiliary constructions (extra lines) added to help solve the problem

Instructions:
1. Look at the auxiliary diagram to see what constructions were added (perpendiculars, connecting segments, midpoints, etc.)
2. Identify the geometric relationships established by these auxiliary lines
3. Use these constructions to set up your solution approach
4. Apply relevant theorems (Pythagorean theorem in 3D, properties of perpendiculars, volume formulas, etc.)
5. Show your step-by-step solution with clear calculations
6. State the final numerical answer

Solve this problem step by step using the auxiliary constructions."""

    return f"[GEN_PROMPT]{generation_prompt}[/GEN_PROMPT]\n[QUESTION]{question_prompt}[/QUESTION]"


def auxsolidmath_process_results(doc: Dict, results: List[str]) -> Dict[str, float]:
    """
    Process auxsolidmath results with LLM Judge evaluation using Azure TRAPI.

    Uses GPT-4o as Judge (configurable via JUDGE_DEPLOYMENT env var).
    """
    result_text = results[0] if results else ""

    # Truncate to last 50000 characters to avoid token limit issues
    # Keep the end since answers are usually at the end
    MAX_RESULT_CHARS = 50000
    if len(result_text) > MAX_RESULT_CHARS:
        result_text = "...[truncated]...\n" + result_text[-MAX_RESULT_CHARS:]

    # Default scores
    text_acc = 0.0

    # Get Judge client
    try:
        judge_client = _get_judge_client()
    except Exception as e:
        print(f"Warning: Failed to initialize Judge client: {e}")
        return {
            "auxsolidmath_text_acc": text_acc,
        }

    # === Text Evaluation ===
    question = doc.get("question", "")
    gt_answer = doc.get("answer", "")
    gt_auxiliary = doc.get("auxiliary_line_description", "")

    if question and gt_answer and result_text:
        text_system = """You are a rigorous grader for solid geometry reasoning.
Given: problem statement (text), ground-truth answer, ground-truth auxiliary line description, and a candidate solution text.

Decide two things:
  (i) is the reasoning rigorous (no major logical gaps or false claims),
  (ii) is the final conclusion correct.

For CALCULATION problems: conclusion correctness means the final NUMERIC result matches the ground-truth
(ignore formatting/units; radicals/π are okay if numerically equivalent).

For PROVING problems: conclusion correctness means the claim is indeed established (may differ in steps but must be valid).

Only if (rigorous AND correct) → text_ok=1, else 0.

Output MUST be a compact JSON: {"reasoning_rigorous":0|1,"conclusion_correct":0|1,"text_ok":0|1,"text_reason":"<short>"}"""

        text_user = f"""Problem:
{question}

Ground truth auxiliary lines:
{gt_auxiliary}

Ground truth answer:
{gt_answer}

Candidate solution:
{result_text}

Evaluate the candidate. Output JSON only."""

        for attempt in range(3):
            try:
                response_text = judge_client.chat_completion(
                    messages=[
                        {"role": "system", "content": text_system},
                        {"role": "user", "content": text_user},
                    ],
                    temperature=0.0,
                    max_tokens=512,
                )
                result_json = _find_first_json_substring(response_text)
                if result_json:
                    data = json.loads(result_json)
                    text_acc = 1.0 if int(data.get("text_ok", 0)) == 1 else 0.0
                break
            except Exception as e:
                print(f"Text Judge attempt {attempt + 1}/3 failed: {e}")
                if attempt < 2:
                    time.sleep(5)

    return {
        "auxsolidmath_text_acc": text_acc,
    }


def auxsolidmath_aggregate(results: List[Optional[float]]) -> float:
    """Aggregate results"""
    vals = [v for v in results if v is not None]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


# ======================================================================
# babyvision
# ======================================================================

"""
BabyVision Task Utilities
Evaluation for Fine-grained Discrimination and Visual Tracking tasks.
Uses Azure TRAPI GPT-4o for LLM Judge evaluation.
"""








LLM_JUDGE_PROMPT = """You are a careful and strict evaluator. You will be given:

1. **Question**
2. **Ground Truth Answer** (correct answer)
3. **Model Output** (answer from another model)

**Your goal:** Determine if the Model Output **accurately matches** the Ground Truth Answer in meaning.

* Matching means: the facts, entities, and key details are equivalent, even if phrasing differs.
* Not matching means: the Model Output is wrong, incomplete, contains extra incorrect facts, or changes the meaning.

**Important:**
* Think through your decision step-by-step **internally** before responding.
* In your final output, return **only** True or False, with no extra text or explanation.

**Input:**

Question: {question}
Ground Truth Answer: {groundtruth}
Model Output: {modeloutput}
"""


def call_judge(question: str, groundtruth: str, modeloutput: str) -> bool:
    """Call LLM Judge to evaluate answer correctness."""
    client = get_judge_client()

    prompt = LLM_JUDGE_PROMPT.format(
        question=question,
        groundtruth=groundtruth,
        modeloutput=modeloutput,
    )

    try:
        response_text = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=16,
            temperature=0,
        ).strip().lower()
        return "true" in response_text
    except Exception as e:
        eval_logger.error(f"[LLM Judge Error] {e}")
        return False


# ============================================================================
# Document Processing Functions
# ============================================================================

def format_choices(choices: List[str]) -> str:
    """Format multiple choice options as (A), (B), (C), etc."""
    if not choices:
        return ""
    formatted = ""
    for idx, choice in enumerate(choices):
        formatted += f"({chr(65 + idx)}) {choice}\n"
    return formatted.strip()


def babyvision_doc_to_visual(doc: Dict) -> List[Image.Image]:
    """Get visual input from document."""
    if "image" in doc and doc["image"]:
        img = doc["image"]
        if isinstance(img, Image.Image):
            return [img.convert("RGB")]
    return []


def babyvision_doc_to_text(doc: Dict, lmms_eval_specific_kwargs: Dict = None) -> str:
    """Format question with choices if applicable."""
    question = doc["question"].strip()

    # Add choices for multiple choice questions
    if doc["ansType"] == "choice" and doc.get("options"):
        question = question + "\nChoices:\n" + format_choices(doc["options"])

    # Add pre_prompt and post_prompt
    if lmms_eval_specific_kwargs:
        if lmms_eval_specific_kwargs.get("pre_prompt"):
            question = lmms_eval_specific_kwargs["pre_prompt"] + question
        if lmms_eval_specific_kwargs.get("post_prompt"):
            question = question + lmms_eval_specific_kwargs["post_prompt"]

    return question


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{} format."""
    if not text:
        return None

    # Match \boxed{...} with nested braces support
    pattern = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{.*\})*\})*)\}"
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()

    # Try backtick format
    pattern_backtick = r"`([^`]+)`"
    matches_bt = re.findall(pattern_backtick, text)
    if matches_bt:
        return matches_bt[-1].strip()

    return None


def babyvision_process_results(doc: Dict, results: List[str]) -> Dict[str, Any]:
    """Process results using Azure TRAPI LLM Judge for evaluation."""
    pred_text = results[0] if results else ""

    # Get ground truth
    if doc["ansType"] == "choice":
        gt_answer = chr(65 + int(doc["choiceAns"])) if doc["choiceAns"] is not None else ""
    else:
        gt_answer = str(doc.get("blankAns", ""))

    # Extract predicted answer from boxed format
    pred_answer = extract_boxed_answer(pred_text)
    if pred_answer is None:
        # Try to find letter answer for choice questions
        if doc["ansType"] == "choice":
            letter_match = re.search(r"\b([A-D])\b", pred_text)
            if letter_match:
                pred_answer = letter_match.group(1)
            else:
                pred_answer = pred_text.strip()[:200]
        else:
            pred_answer = pred_text.strip()[:200]

    # Use LLM Judge for evaluation
    is_correct = call_judge(
        question=doc["question"],
        groundtruth=gt_answer,
        modeloutput=pred_answer,
    )
    score = 1.0 if is_correct else 0.0

    task_type = doc.get("type", "unknown")
    subtype = doc.get("subtype", "unknown")

    return {
        task_type: {
            "task_id": doc.get("taskId"),
            "subtype": subtype,
            "score": score,
            "gt": gt_answer,
            "pred": pred_answer,
        },
        "accuracy": {
            "task_id": doc.get("taskId"),
            "subtype": subtype,
            "score": score,
        },
    }


def babyvision_aggregate_results(results: List[Dict]) -> float:
    """Aggregate results to compute accuracy."""
    subtype_scores = defaultdict(list)

    for result in results:
        score = result["score"]
        subtype = result["subtype"]
        subtype_scores[subtype].append(score)

    # Log subtype accuracies
    for subtype, scores in sorted(subtype_scores.items()):
        avg = sum(scores) / len(scores) if scores else 0.0
        eval_logger.info(f"  {subtype}: {avg:.4f} ({sum(scores):.0f}/{len(scores)})")

    # Overall accuracy
    all_scores = [s for scores in subtype_scores.values() for s in scores]
    return sum(all_scores) / len(all_scores) if all_scores else 0.0


# ============================================================================
# Dataset filtering functions (used by process_docs)
# ============================================================================

def process_docs(dataset, task_type: str):
    """Filter dataset by task type."""
    return dataset.filter(lambda x: x["type"] == task_type)


process_fine_grained = partial(process_docs, task_type="Fine-grained Discrimination")
process_visual_tracking = partial(process_docs, task_type="Visual Tracking")


# ============================================================================
# Visual CoT Versions
# ============================================================================

# Fine-grained Discrimination: 需要辨别细微差异，生成图突出细节
FINE_GRAINED_GEN_PROMPT = (
    "This image contains subtle visual details that need careful examination. "
    "Your task: Generate a visualization that highlights and emphasizes the fine-grained details, "
    "subtle differences, and distinguishing features in the image. "
    "Make the key discriminative elements more prominent and easier to identify."
)

# Visual Tracking: 需要追踪物体，生成图突出物体轨迹/位置
VISUAL_TRACKING_GEN_PROMPT = (
    "This image involves tracking objects or their movements. "
    "Your task: Generate a visualization that highlights the objects of interest, "
    "their positions, trajectories, or movement patterns. "
    "Make it easier to follow and track the relevant visual elements."
)


def doc_to_text_visual_cot(
    doc: Dict,
    lmms_eval_specific_kwargs: Dict = None,
    gen_prompt: str = "",
) -> str:
    """Format Visual CoT prompt with generation prompt and question."""
    question = doc["question"].strip()

    # Add choices for multiple choice questions
    if doc["ansType"] == "choice" and doc.get("options"):
        question = question + "\nChoices:\n" + format_choices(doc["options"])

    # Add auxiliary image notice
    question = (
        "In addition to the original image, you are also given an auxiliary "
        "visualization image that highlights key visual elements.\n\n" + question
    )

    # Add pre_prompt and post_prompt
    if lmms_eval_specific_kwargs:
        if lmms_eval_specific_kwargs.get("pre_prompt"):
            question = lmms_eval_specific_kwargs["pre_prompt"] + question
        if lmms_eval_specific_kwargs.get("post_prompt"):
            question = question + lmms_eval_specific_kwargs["post_prompt"]

    return f"[GEN_PROMPT]{gen_prompt}[/GEN_PROMPT][QUESTION]{question}[/QUESTION]"


def doc_to_text_fine_grained_cot(
    doc: Dict, lmms_eval_specific_kwargs: Dict = None
) -> str:
    """Visual CoT prompt for Fine-grained Discrimination task."""
    return doc_to_text_visual_cot(
        doc, lmms_eval_specific_kwargs, FINE_GRAINED_GEN_PROMPT
    )


def doc_to_text_visual_tracking_cot(
    doc: Dict, lmms_eval_specific_kwargs: Dict = None
) -> str:
    """Visual CoT prompt for Visual Tracking task."""
    return doc_to_text_visual_cot(
        doc, lmms_eval_specific_kwargs, VISUAL_TRACKING_GEN_PROMPT
    )


# ======================================================================
# geometry3k
# ======================================================================

"""
Geometry3K Task Utilities
Evaluation for plane geometry problems from the Geometry3K dataset.
"""







def _find_first_json_substring(text: str) -> Optional[str]:
    """Extract first JSON object from text"""
    if not text:
        return None
    start_index = text.find("{")
    if start_index == -1:
        return None

    brace_depth, in_string, is_escaped = 0, False, False
    for i in range(start_index, len(text)):
        char = text[i]
        if char == '"' and not is_escaped:
            in_string = not in_string
        if not in_string:
            if char == "{":
                brace_depth += 1
            elif char == "}":
                brace_depth -= 1
                if brace_depth == 0:
                    return text[start_index : i + 1]
        is_escaped = char == "\\" and not is_escaped
    return None


def geometry3k_doc_to_visual(doc: Dict) -> List[Image.Image]:
    """Get visual input for geometry3k task"""
    images = doc.get("images", [])
    if images:
        # images is a list, return the first image
        if isinstance(images, list) and len(images) > 0:
            img = images[0]
            if isinstance(img, Image.Image):
                return [img]
    return []


def geometry3k_doc_to_text(doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None) -> str:
    """Get text prompt for geometry3k task"""
    problem = doc.get("problem", "")

    # Problem already contains <image> tag, so we use it directly
    return f"""{problem}

Instructions:
1. Carefully analyze the geometry diagram shown above.
2. Read the problem statement and identify what needs to be found.
3. Show your step-by-step solution with clear reasoning.
4. Include all intermediate calculations.
5. State the final answer clearly at the end.

Please solve this problem step by step."""


def geometry3k_doc_to_target(doc: Dict) -> str:
    """Get target answer for geometry3k task"""
    return doc.get("answer", "")


def geometry3k_doc_to_text_visual_cot(
    doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None
) -> str:
    """
    Get two-stage Visual Chain-of-Thought prompt for geometry3k task.

    Stage 1: Analyze the problem and draw auxiliary lines on the original diagram
    Stage 2: Solve the problem using both original diagram and the auxiliary diagram
    """
    problem = doc.get("problem", "")

    # Stage 1: Analyze problem and generate diagram with auxiliary constructions
    generation_prompt = f"""You are given a geometry problem with a diagram. Analyze the problem and create an enhanced version of the SAME diagram with auxiliary constructions added.

Problem: {problem}

Instructions:
1. KEEP all original elements exactly as they are (all points, lines, labels, and measurements)
2. Analyze what auxiliary constructions would help solve this problem
3. ADD auxiliary lines in a different color (e.g., red or dashed lines):
   - Perpendicular lines from center to chords
   - Extended lines if needed
   - Angle bisectors, midpoints, or other helpful constructions
4. Label any new points you add (use letters not already in the diagram)
5. The final diagram should look like the original with extra auxiliary lines drawn on top

Generate an enhanced diagram that preserves the original and adds helpful auxiliary constructions."""

    # Stage 2: Solve using both original and auxiliary diagram
    question_prompt = f"""{problem}

You are given TWO images:
1) ORIGINAL DIAGRAM: The geometry problem as given
2) AUXILIARY DIAGRAM: The same diagram with auxiliary constructions (extra lines) added to help solve the problem

Instructions:
1. Look at the auxiliary diagram to see what constructions were added
2. Use these auxiliary lines to identify key geometric relationships (perpendiculars, congruent segments, etc.)
3. Apply relevant theorems (Pythagorean theorem, chord properties, etc.)
4. Show your step-by-step solution with clear calculations
5. State the final numerical answer

Solve this problem step by step."""

    return f"[GEN_PROMPT]{generation_prompt}[/GEN_PROMPT]\n[QUESTION]{question_prompt}[/QUESTION]"


def geometry3k_process_results(doc: Dict, results: List[str]) -> Dict[str, float]:
    """
    Process geometry3k results with LLM Judge evaluation using Azure TRAPI.

    Uses GPT-4o as Judge (configurable via JUDGE_DEPLOYMENT env var).
    """
    result_text = results[0] if results else ""

    # Truncate to last 50000 characters to avoid token limit issues
    # Keep the end since answers are usually at the end
    MAX_RESULT_CHARS = 50000
    if len(result_text) > MAX_RESULT_CHARS:
        result_text = "...[truncated]...\n" + result_text[-MAX_RESULT_CHARS:]

    # Default scores
    accuracy = 0.0

    # Get Judge client
    try:
        judge_client = _get_judge_client()
    except Exception as e:
        print(f"Warning: Failed to initialize Judge client: {e}")
        return {
            "geometry3k_accuracy": accuracy,
        }

    # === Answer Evaluation ===
    problem = doc.get("problem", "")
    gt_answer = doc.get("answer", "")

    if problem and gt_answer and result_text:
        judge_system = """You are a rigorous grader for plane geometry problems.
Given: problem statement (text with diagram), ground-truth answer, and a candidate solution text.

Evaluate if the candidate's final answer is mathematically equivalent to the ground-truth answer.

Important considerations:
- Numeric answers: Check if values match (allow minor rounding differences, e.g., 9.2 vs 9.19)
- Algebraic expressions: Check mathematical equivalence (e.g., "2√221" = "2*sqrt(221)")
- Fractions: Check equivalence (e.g., "1/2" = "0.5")
- LaTeX formatting: Ignore formatting differences (e.g., "\frac{1}{2}" = "1/2")
- Angle measures: Check numeric equivalence
- Units: Ignore unit differences if values match

Only mark as correct (answer_correct=1) if the final answer is mathematically equivalent.
Mark as incorrect (answer_correct=0) if:
- The answer is wrong
- No clear final answer is provided
- The answer is ambiguous

Output MUST be a compact JSON: {"answer_correct":0|1,"reason":"<short explanation>"}"""

        judge_user = f"""Problem:
{problem}

Ground truth answer:
{gt_answer}

Candidate solution:
{result_text}

Evaluate if the candidate's final answer matches the ground truth. Output JSON only."""

        for attempt in range(3):
            try:
                response_text = judge_client.chat_completion(
                    messages=[
                        {"role": "system", "content": judge_system},
                        {"role": "user", "content": judge_user},
                    ],
                    temperature=0.0,
                    max_tokens=512,
                )
                result_json = _find_first_json_substring(response_text)
                if result_json:
                    data = json.loads(result_json)
                    accuracy = 1.0 if int(data.get("answer_correct", 0)) == 1 else 0.0
                break
            except Exception as e:
                print(f"Judge attempt {attempt + 1}/3 failed: {e}")
                if attempt < 2:
                    time.sleep(5)

    return {
        "geometry3k_accuracy": accuracy,
    }


def geometry3k_aggregate(results: List[Optional[float]]) -> float:
    """Aggregate results"""
    vals = [v for v in results if v is not None]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


# ======================================================================
# phyx
# ======================================================================



LLM_JUDGE_PROMPT = """You are a careful and strict evaluator. You will be given:

1. **Question**
2. **Ground Truth Answer** (correct answer)
3. **Model Output** (answer from another model)

**Your goal:** Determine if the Model Output **accurately matches** the Ground Truth Answer in meaning.

* Matching means: the facts, entities, and key details are equivalent, even if phrasing differs.
* Not matching means: the Model Output is wrong, incomplete, contains extra incorrect facts, or changes the meaning.

**Important:**
* Think through your decision step-by-step **internally** before responding.
* In your final output, return **only** True or False, with no extra text or explanation.

**Input:**

Question: {question}
Ground Truth Answer: {groundtruth}
Model Output: {modeloutput}
"""


def call_judge(question: str, groundtruth: str, modeloutput: str) -> bool:
    """Call LLM Judge to evaluate answer correctness."""
    client = get_judge_client()

    prompt = LLM_JUDGE_PROMPT.format(
        question=question,
        groundtruth=groundtruth,
        modeloutput=modeloutput,
    )

    try:
        response_text = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=16,
            temperature=0,
        ).strip().lower()
        return "true" in response_text
    except Exception as e:
        eval_logger.error(f"[LLM Judge Error] {e}")
        return False


def load_phyx_config():
    config_path = Path(__file__).parent / "phyx.yaml"
    if not config_path.exists():
        # Try parent phyx task directory
        config_path = Path(__file__).parent.parent / "phyx" / "phyx.yaml"
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        raw_data = f.readlines()
        safe_data = []
        for line in raw_data:
            if "!function" not in line:
                safe_data.append(line)
        return yaml.safe_load("".join(safe_data))


config = load_phyx_config()


def decode_base64_to_image(base64_string, target_size=-1):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    if target_size > 0:
        image.thumbnail((target_size, target_size))
    return image


def phyx_doc_to_visual(doc):
    image = decode_base64_to_image(doc["image"])
    return [image]


def phyx_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    query_prompt = doc["question"]
    return query_prompt


def extract_boxed_answer(text: str):
    """Extract answer from \\boxed{} format."""
    if not text:
        return None

    # Match \boxed{...} with nested braces support
    pattern = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{.*\})*\})*)\}"
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()

    # Try backtick format
    pattern_backtick = r"`([^`]+)`"
    matches_bt = re.findall(pattern_backtick, text)
    if matches_bt:
        return matches_bt[-1].strip()

    return None


def phyx_process_results_mc(doc, results):
    """Process multiple choice results using Azure TRAPI LLM Judge."""
    prediction = results[0].strip() if results else ""
    
    # Extract answer from prediction
    pred_answer = extract_boxed_answer(prediction)
    if pred_answer is None:
        # Try to find letter answer
        letter_match = re.search(r"\b([A-Z])\b", prediction)
        if letter_match:
            pred_answer = letter_match.group(1)
        else:
            pred_answer = prediction[:200]
    
    # Get ground truth
    gt_answer = str(doc["answer"])
    
    # Use LLM Judge for evaluation
    is_correct = call_judge(
        question=doc["question"],
        groundtruth=gt_answer,
        modeloutput=pred_answer,
    )
    
    eval_result = {
        "index": doc["index"],
        "true_false": is_correct,
        "category": doc["category"],
        "answer": doc["answer"],
    }

    return {
        "eval_results": eval_result,
    }


def phyx_process_results(doc, results):
    """Process results using Azure TRAPI LLM Judge."""
    prediction = results[0].strip() if results else ""
    
    # Extract answer from prediction
    pred_answer = extract_boxed_answer(prediction)
    if pred_answer is None:
        pred_answer = prediction[:200]
    
    # Get ground truth
    gt_answer = str(doc["answer"])
    
    # Use LLM Judge for evaluation
    is_correct = call_judge(
        question=doc["question"],
        groundtruth=gt_answer,
        modeloutput=pred_answer,
    )
    
    eval_result = {
        "index": doc["index"],
        "true_false": is_correct,
        "category": doc["category"],
        "answer": doc["answer"],
    }

    return {
        "eval_results": eval_result,
    }


def phyx_aggregate_results(results):
    hit = [x["true_false"] for x in results]
    Overall_acc = np.mean(hit)
    return Overall_acc


# ============================================================================
# Visual CoT Versions
# ============================================================================

# Optics: 根据题目画光路图
OPTICS_GEN_PROMPT = (
    "Based on this optics problem, draw a light ray diagram that helps solve the problem. "
    "Show the paths of light rays, including incident rays, reflected rays, refracted rays, "
    "and any relevant angles or focal points as needed by the problem."
)

# Mechanics: 根据题目画受力分析图
MECHANICS_GEN_PROMPT = (
    "Based on this mechanics problem, draw a free body diagram (force analysis diagram) "
    "that helps solve the problem. "
    "Show all the forces acting on the object(s), including gravity, normal force, friction, "
    "tension, applied forces, etc., with arrows indicating direction and relative magnitude."
)


def phyx_doc_to_text_optics_cot(doc, lmms_eval_specific_kwargs=None):
    """Visual CoT prompt for PhyX Optics task."""
    question = (
        "In addition to the original image, you are also given an auxiliary "
        "light ray diagram to help you solve the problem.\n\n"
        + doc["question"]
    )
    return f"[GEN_PROMPT]{OPTICS_GEN_PROMPT}[/GEN_PROMPT][QUESTION]{question}[/QUESTION]"


def phyx_doc_to_text_mechanics_cot(doc, lmms_eval_specific_kwargs=None):
    """Visual CoT prompt for PhyX Mechanics task."""
    question = (
        "In addition to the original image, you are also given an auxiliary "
        "free body diagram (force analysis diagram) to help you solve the problem.\n\n"
        + doc["question"]
    )
    return f"[GEN_PROMPT]{MECHANICS_GEN_PROMPT}[/GEN_PROMPT][QUESTION]{question}[/QUESTION]"


# ======================================================================
# realunify
# ======================================================================

"""
RealUnify Task Utilities
Evaluation for GEU (Generation Enhances Understanding) tasks:
- Mental Tracking
- Mental Reconstruction
- Attentional Focusing
"""




def realunify_doc_to_visual(doc: Dict) -> List[Image.Image]:
    """Get visual input from document."""
    if "image" in doc and doc["image"]:
        img_data = doc["image"]
        # Handle different image formats
        if isinstance(img_data, Image.Image):
            return [img_data.convert("RGB")]
        elif isinstance(img_data, bytes):
            return [Image.open(BytesIO(img_data)).convert("RGB")]
        elif isinstance(img_data, dict) and "bytes" in img_data:
            return [Image.open(BytesIO(img_data["bytes"])).convert("RGB")]
    return []


def realunify_doc_to_text(doc: Dict, lmms_eval_specific_kwargs: Dict = None) -> str:
    """Format evaluation prompt."""
    # GEU tasks use evaluation_prompt field
    prompt = doc.get("evaluation_prompt", doc.get("prompt", "")).strip()

    # Add pre_prompt and post_prompt
    if lmms_eval_specific_kwargs:
        if lmms_eval_specific_kwargs.get("pre_prompt"):
            prompt = lmms_eval_specific_kwargs["pre_prompt"] + prompt
        if lmms_eval_specific_kwargs.get("post_prompt"):
            prompt = prompt + lmms_eval_specific_kwargs["post_prompt"]

    return prompt


def extract_answer(response: str) -> str:
    """Extract answer letter from model response."""
    response = response.strip()

    # Remove common answer prefixes
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
        "Answer",
        "Answer is",
    ]
    for prefix in answer_prefixes:
        response = response.replace(prefix, "")

    # If response is too long and no clear answer, return empty
    if len(response.split()) > 10 and not re.search("[ABCD]", response):
        return ""

    # Find first A/B/C/D
    match = re.search(r"[ABCD]", response)
    if match:
        return match.group(0)
    return ""


def realunify_process_results(doc: Dict, results: List[str]) -> Dict[str, Any]:
    """Process results - extract answer and compare to ground truth."""
    pred_text = results[0] if results else ""

    # Extract predicted answer
    pred_answer = extract_answer(pred_text)

    # Get ground truth
    gt_answer = doc.get("answer", "")

    # Compare
    score = 1.0 if pred_answer == gt_answer else 0.0

    task_type = doc.get("task_type", "unknown")

    return {
        task_type: {
            "score": score,
            "gt": gt_answer,
            "pred": pred_answer,
        },
        "accuracy": {
            "task_type": task_type,
            "score": score,
        },
    }


def realunify_aggregate_results(results: List[Dict]) -> float:
    """Aggregate results to compute accuracy."""
    task_scores = defaultdict(list)

    for result in results:
        score = result["score"]
        task_type = result.get("task_type", "unknown")
        task_scores[task_type].append(score)

    # Log per-task accuracies
    for task_type, scores in sorted(task_scores.items()):
        avg = sum(scores) / len(scores) if scores else 0.0
        eval_logger.info(f"  {task_type}: {avg:.4f} ({sum(scores):.0f}/{len(scores)})")

    # Overall accuracy
    all_scores = [s for scores in task_scores.values() for s in scores]
    return sum(all_scores) / len(all_scores) if all_scores else 0.0


# ============================================================================
# Visual CoT Versions
# ============================================================================

# Mental Reconstruction: 图片被打乱，需要恢复
MENTAL_RECONSTRUCTION_GEN_PROMPT = (
    "Please restore the image that has been shuffled by patches, "
    "without adding extra content or altering the original image."
)

# Attentional Focusing: 高亮与问题相关的区域
ATTENTIONAL_FOCUSING_GEN_PROMPT_TEMPLATE = (
    "Here is the question: {question}\n"
    "Please highlight the regions of the image that are relevant to the question."
)

# Mental Tracking: 根据问题对图片内容进行变换
MENTAL_TRACKING_GEN_PROMPT_TEMPLATE = (
    "Here is the question: {question}\n"
    "Please apply the corresponding transformations and modifications "
    "to the contents of the image according to the question."
)


def doc_to_text_mental_reconstruction_cot(
    doc: Dict, lmms_eval_specific_kwargs: Dict = None
) -> str:
    """Visual CoT prompt for Mental Reconstruction task."""
    prompt = doc.get("evaluation_prompt", doc.get("prompt", "")).strip()

    question_with_aux = (
        "In addition to the original image, you are also given a restored version "
        "of the shuffled image to help you answer the question.\n\n"
        + prompt
    )

    # Add pre_prompt and post_prompt
    if lmms_eval_specific_kwargs:
        if lmms_eval_specific_kwargs.get("pre_prompt"):
            question_with_aux = (
                lmms_eval_specific_kwargs["pre_prompt"] + question_with_aux
            )
        if lmms_eval_specific_kwargs.get("post_prompt"):
            question_with_aux = (
                question_with_aux + lmms_eval_specific_kwargs["post_prompt"]
            )

    return (
        f"[GEN_PROMPT]{MENTAL_RECONSTRUCTION_GEN_PROMPT}[/GEN_PROMPT]"
        f"[QUESTION]{question_with_aux}[/QUESTION]"
    )


def doc_to_text_attentional_focusing_cot(
    doc: Dict, lmms_eval_specific_kwargs: Dict = None
) -> str:
    """Visual CoT prompt for Attentional Focusing task."""
    question = doc.get("question", "").strip()
    prompt = doc.get("evaluation_prompt", doc.get("prompt", "")).strip()

    gen_prompt = ATTENTIONAL_FOCUSING_GEN_PROMPT_TEMPLATE.format(question=question)

    question_with_aux = (
        "In addition to the original image, you are also given a visualization "
        "that highlights the regions relevant to the question.\n\n"
        + prompt
    )

    # Add pre_prompt and post_prompt
    if lmms_eval_specific_kwargs:
        if lmms_eval_specific_kwargs.get("pre_prompt"):
            question_with_aux = (
                lmms_eval_specific_kwargs["pre_prompt"] + question_with_aux
            )
        if lmms_eval_specific_kwargs.get("post_prompt"):
            question_with_aux = (
                question_with_aux + lmms_eval_specific_kwargs["post_prompt"]
            )

    return f"[GEN_PROMPT]{gen_prompt}[/GEN_PROMPT][QUESTION]{question_with_aux}[/QUESTION]"


def doc_to_text_mental_tracking_cot(
    doc: Dict, lmms_eval_specific_kwargs: Dict = None
) -> str:
    """Visual CoT prompt for Mental Tracking task."""
    question = doc.get("question", "").strip()
    prompt = doc.get("evaluation_prompt", doc.get("prompt", "")).strip()

    gen_prompt = MENTAL_TRACKING_GEN_PROMPT_TEMPLATE.format(question=question)

    question_with_aux = (
        "In addition to the original image, you are also given a transformed version "
        "of the image with the modifications applied according to the question.\n\n"
        + prompt
    )

    # Add pre_prompt and post_prompt
    if lmms_eval_specific_kwargs:
        if lmms_eval_specific_kwargs.get("pre_prompt"):
            question_with_aux = (
                lmms_eval_specific_kwargs["pre_prompt"] + question_with_aux
            )
        if lmms_eval_specific_kwargs.get("post_prompt"):
            question_with_aux = (
                question_with_aux + lmms_eval_specific_kwargs["post_prompt"]
            )

    return f"[GEN_PROMPT]{gen_prompt}[/GEN_PROMPT][QUESTION]{question_with_aux}[/QUESTION]"


# ======================================================================
# uni_mmmu
# ======================================================================

"""
Uni-MMMU Task Utilities
Text-only evaluation using GPT-4o API via Azure TRAPI.
"""



# Azure OpenAI imports



# ============================================================================
# GPT-4o API Client (from api.py)
# ============================================================================

_CLIENT = None
_DEPLOYMENT = None


def get_gpt4o_client():
    """Get or create Azure OpenAI client for GPT-4o."""
    global _CLIENT, _DEPLOYMENT

    if _CLIENT is None:
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if azure_endpoint:
            token = _get_azure_cli_token()
            _DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
            _CLIENT = AzureOpenAI(
                azure_endpoint=azure_endpoint.replace("/openai/v1", ""),
                api_key=token,
                api_version=api_version,
            )
        elif has_endpoint_support():
            _CLIENT, _DEPLOYMENT = build_azure_compat_client()
        elif AZURE_AVAILABLE:
            scope = os.getenv("TRAPI_SCOPE", "api://trapi/.default")
            api_version = os.getenv("TRAPI_API_VERSION", "2024-10-21")
            _DEPLOYMENT = os.getenv("TRAPI_DEPLOYMENT", "gpt-4o_2024-11-20")
            instance = os.getenv("TRAPI_INSTANCE", "gcr/shared")
            endpoint = f"https://trapi.research.microsoft.com/{instance}"

            chained = ChainedTokenCredential(
                AzureCliCredential(),
                ManagedIdentityCredential(),
            )
            credential_provider = get_bearer_token_provider(chained, scope)

            _CLIENT = AzureOpenAI(
                azure_endpoint=endpoint,
                azure_ad_token_provider=credential_provider,
                api_version=api_version,
            )
        else:
            raise RuntimeError("No Azure/OpenAI backend available for GPT-4o client")

    return _CLIENT, _DEPLOYMENT


def call_gpt4o(prompt: str, max_tokens: int = 512) -> str:
    """Call GPT-4o API with a text prompt."""
    client, deployment = get_gpt4o_client()

    try:
        resp = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0,
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"[GPT-4o Error] {e}")
        return ""


# ============================================================================
# Common Utilities
# ============================================================================

def find_first_json_substring(text: str) -> Optional[str]:
    """Extract first JSON object from text."""
    if not text:
        return None
    start_index = text.find("{")
    if start_index == -1:
        return None

    brace_depth, in_string, is_escaped = 0, False, False
    for i in range(start_index, len(text)):
        char = text[i]
        if char == '"' and not is_escaped:
            in_string = not in_string
        if not in_string:
            if char == "{":
                brace_depth += 1
            elif char == "}":
                brace_depth -= 1
                if brace_depth == 0:
                    return text[start_index : i + 1]
        is_escaped = char == "\\" and not is_escaped
    return None


# ============================================================================
# Jigsaw Task Functions
# ============================================================================

def jigsaw_doc_to_visual(doc: Dict) -> List[Image.Image]:
    """Get visual inputs for jigsaw task."""
    images = []
    for key in ["ref_image", "cand0_image", "cand1_image"]:
        if key in doc and doc[key]:
            img_bytes = doc[key]["bytes"] if isinstance(doc[key], dict) else doc[key]
            images.append(Image.open(BytesIO(img_bytes)).convert("RGB"))
    return images


def jigsaw_doc_to_text(doc: Dict, lmms_eval_specific_kwargs: Dict = None) -> str:
    """Get text prompt for jigsaw task - text-only evaluation."""
    prompt = """You are a unified vision-language model. You will be given:
(1) a 2×2 reference image with the bottom-right cell hidden, and
(2) two candidate patch images ("Candidate 0" and "Candidate 1").

Your job:
- Mentally visualize placing each candidate into the bottom-right cell.
- Compare which candidate yields the correct completion based on seam continuity, color/texture gradient, structural alignment, and global semantics.

Output EXACTLY the following:

1) Brief analysis comparing the two candidates

2) One strict JSON object with your decision, wrapped as:
<FINAL_ANSWER_JSON>
{"choice": 0 or 1, "rationale": "≤30 words decisive cue"}
</FINAL_ANSWER_JSON>

Hard constraints:
- Deterministic, single outcome. No hedging, no multiple possibilities.
- Do not restate the task as the answer; reason from visual evidence.

Inputs:"""
    return prompt


def jigsaw_process_results(doc: Dict, results: List[str]) -> Dict[str, float]:
    """Process jigsaw results - text evaluation only."""
    result_raw = results[0] if results else ""
    
    # Handle case where result is a JSON string (from bagel format_output)
    result_text = ""
    if isinstance(result_raw, str):
        # Try to parse as JSON first (bagel outputs JSON string)
        try:
            parsed_result = json.loads(result_raw)
            if isinstance(parsed_result, dict) and "text" in parsed_result:
                result_text = parsed_result["text"]
            else:
                result_text = result_raw
        except (json.JSONDecodeError, TypeError):
            # Not JSON, use as-is
            result_text = result_raw
    else:
        result_text = str(result_raw)

    # Parse choice from <FINAL_ANSWER_JSON> or <FINAL_ANSWER JSON> (flexible matching)
    choice = None
    
    # Try exact match first: <FINAL_ANSWER_JSON>
    match = re.search(
        r"<FINAL_ANSWER_JSON>\s*(\{.*?\})\s*</FINAL_ANSWER_JSON>",
        result_text,
        re.DOTALL | re.IGNORECASE,
    )
    
    # If not found, try with space: <FINAL_ANSWER JSON>
    if not match:
        match = re.search(
            r"<FINAL_ANSWER\s+JSON>\s*(\{.*?\})\s*</FINAL_ANSWER\s+JSON>",
            result_text,
            re.DOTALL | re.IGNORECASE,
        )
    
    if match:
        json_str = match.group(1)
        parsed = find_first_json_substring(json_str)
        if parsed:
            try:
                data = json.loads(parsed)
                choice = data.get("choice")
            except (json.JSONDecodeError, ValueError, KeyError):
                pass

    if choice is None:
        choice_match = re.search(r'"choice"\s*:\s*(\d)', result_text)
        if choice_match:
            choice = int(choice_match.group(1))

    gt_label = doc.get("label", -1)
    if isinstance(gt_label, str):
        try:
            gt_label = int(gt_label)
        except ValueError:
            gt_label = -1
    
    if choice is not None and not isinstance(choice, int):
        try:
            choice = int(choice)
        except (ValueError, TypeError):
            choice = None
    
    text_correct = 1 if choice is not None and choice == gt_label else 0

    return {"jigsaw_text_acc": text_correct}


# ============================================================================
# Maze Task Functions
# ============================================================================

def maze_doc_to_visual(doc: Dict) -> List[Image.Image]:
    """Get visual input for maze task."""
    if "initial_image" in doc and doc["initial_image"]:
        img_bytes = doc["initial_image"]["bytes"] if isinstance(doc["initial_image"], dict) else doc["initial_image"]
        return [Image.open(BytesIO(img_bytes)).convert("RGB")]
    return []


def maze_doc_to_text(doc: Dict, lmms_eval_specific_kwargs: Dict = None) -> str:
    """Get text prompt for maze task - text-only evaluation."""
    prompt = """You are a precise maze solver.

SEMANTICS
- Black squares: walls (impassable)
- White squares: path (walkable)
- Blue dot: start (the agent)
- Green rectangular frame: goal (reaching any white cell inside the green frame counts as success)
- Legal moves: up, down, left, right only. One cell per step; no diagonals, no jumps; never cross walls.

OUTPUT FORMAT
1) Briefly describe your reasoning for the path.

2) Output the final move list as a JSON array of lowercase strings, wrapped as:
<ANSWER_JSON>["right","down","left"]</ANSWER_JSON>

NO EXTRAS
- No extra explanations beyond the brief reasoning and the JSON answer."""
    return prompt


def maze_process_results(doc: Dict, results: List[str]) -> Dict[str, float]:
    """Process maze results - text evaluation only using GPT-4o."""
    result_raw = results[0] if results else ""
    
    # Handle case where result is a JSON string (from bagel format_output)
    result_text = ""
    if isinstance(result_raw, str):
        # Try to parse as JSON first (bagel outputs JSON string)
        try:
            parsed_result = json.loads(result_raw)
            if isinstance(parsed_result, dict) and "text" in parsed_result:
                result_text = parsed_result["text"]
            else:
                result_text = result_raw
        except (json.JSONDecodeError, TypeError):
            # Not JSON, use as-is
            result_text = result_raw
    else:
        result_text = str(result_raw)

    # Parse predicted moves from <ANSWER_JSON>
    pred_moves = []
    matches = list(re.finditer(
        r"<ANSWER_JSON>\s*(\[.*?\])\s*</ANSWER_JSON>",
        result_text,
        re.DOTALL | re.IGNORECASE
    ))
    if matches:
        try:
            moves_data = json.loads(matches[-1].group(1))
            pred_moves = [str(m).strip().lower() for m in moves_data]
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

    # Ground truth moves
    gt_moves_str = doc.get("steps", "[]")
    gt_moves = json.loads(gt_moves_str) if isinstance(gt_moves_str, str) else gt_moves_str
    gt_moves = [str(m).lower() for m in gt_moves]

    # Text evaluation
    text_exact = 1 if pred_moves == gt_moves else 0
    text_frame_acc = (
        sum(1 for p, g in zip(pred_moves, gt_moves) if p == g) / len(gt_moves)
        if gt_moves else 0.0
    )

    return {
        "maze_text_exact": text_exact,
        "maze_text_frame_acc": text_frame_acc,
    }


# ============================================================================
# Sliding Puzzle Task Functions
# ============================================================================

def sliding_doc_to_visual(doc: Dict) -> List[Image.Image]:
    """Get visual input for sliding puzzle task."""
    if "initial_image" in doc and doc["initial_image"]:
        img_bytes = doc["initial_image"]["bytes"] if isinstance(doc["initial_image"], dict) else doc["initial_image"]
        return [Image.open(BytesIO(img_bytes)).convert("RGB")]
    return []


def sliding_doc_to_text(doc: Dict, lmms_eval_specific_kwargs: Dict = None) -> str:
    """Get text prompt for sliding puzzle task - text-only evaluation."""
    prompt = """You are a precise sliding puzzle solver.

TASK
- You will be given an INITIAL state of a 3x3 sliding puzzle.
- The goal is to find the sequence of moves to solve the puzzle.

SEMANTICS
- The puzzle is a 3x3 grid with 8 colored tiles and one empty space.
- The RED square represents the EMPTY space.
- A "move" consists of sliding an adjacent colored tile INTO the empty (red) space.
- Moves are named by the direction the COLORED TILE moves. For example, if the blue tile is directly above the red space, moving the blue tile down into the red space's position is a "down" move.
- Legal moves: up, down, left, right only. One tile per step.

OUTPUT FORMAT
1) Briefly describe your reasoning for the solution.

2) Output the final move list as a JSON array of lowercase strings, wrapped as:
<ANSWER_JSON>["down","right","up"]</ANSWER_JSON>

NO EXTRAS
- No extra explanations beyond the brief reasoning and the JSON answer."""
    return prompt


def sliding_process_results(doc: Dict, results: List[str]) -> Dict[str, float]:
    """Process sliding puzzle results - text evaluation only."""
    result_raw = results[0] if results else ""
    
    # Handle case where result is a JSON string (from bagel format_output)
    result_text = ""
    if isinstance(result_raw, str):
        # Try to parse as JSON first (bagel outputs JSON string)
        try:
            parsed_result = json.loads(result_raw)
            if isinstance(parsed_result, dict) and "text" in parsed_result:
                result_text = parsed_result["text"]
            else:
                result_text = result_raw
        except (json.JSONDecodeError, TypeError):
            # Not JSON, use as-is
            result_text = result_raw
    else:
        result_text = str(result_raw)

    # Parse predicted moves from <ANSWER_JSON>
    pred_moves = []
    matches = list(re.finditer(
        r"<ANSWER_JSON>\s*(\[.*?\])\s*</ANSWER_JSON>",
        result_text,
        re.DOTALL | re.IGNORECASE
    ))
    if matches:
        try:
            moves_data = json.loads(matches[-1].group(1))
            pred_moves = [str(m).strip().lower() for m in moves_data]
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

    # Ground truth moves
    gt_moves_str = doc.get("steps_words", "[]")
    gt_moves = json.loads(gt_moves_str) if isinstance(gt_moves_str, str) else gt_moves_str
    gt_moves = [str(m).lower() for m in gt_moves]

    # Convert ground truth moves: swap up<->down, left<->right
    # This is needed because the coordinate system in the dataset may differ from model's understanding
    direction_map = {
        "up": "down",
        "down": "up",
        "left": "right",
        "right": "left"
    }
    gt_moves = [direction_map.get(m, m) for m in gt_moves]

    # Text evaluation
    text_exact = 1 if pred_moves == gt_moves else 0
    text_frame_acc = (
        sum(1 for p, g in zip(pred_moves, gt_moves) if p == g) / len(gt_moves)
        if gt_moves else 0.0
    )

    return {
        "sliding_text_exact": text_exact,
        "sliding_text_frame_acc": text_frame_acc,
    }


# ============================================================================
# Visual CoT Versions - Aligned with Original Uni-MMMU
# ============================================================================
# These prompts match the exact prompts used in the original Uni-MMMU benchmark


# Jigsaw prompt (aligned with original jigsaw.py)
JIGSAW_PROMPT = """You are a unified vision-language model. You will be given:
(1) a 2×2 reference image with the bottom-right cell hidden, and
(2) two candidate patch images ("Candidate 0" and "Candidate 1").

Your job:
- For each candidate, synthesize a completed 2×2 image by placing that candidate EXACTLY into the bottom-right cell. Keep the other three cells pixel-identical to the reference (no filtering, no re-rendering). If sizes differ, only scale the candidate to fit that quadrant; do NOT rotate, mirror, or alter colors.
- Compare the two completed results and decide which candidate yields the correct completion.

Output EXACTLY the following, in order:

1) A single image with Candidate 0 placed in the bottom-right cell

2) A single image with Candidate 1 placed in the bottom-right cell


3) analysis comparing seam continuity, color/texture gradient, structural alignment, and global semantics

4) One strict JSON object with your decision, wrapped as:
<FINAL_ANSWER_JSON>
{"choice": 0 or 1, "rationale": "≤30 words decisive cue"}
</FINAL_ANSWER_JSON>

Hard constraints:
- Deterministic, single outcome. No hedging, no multiple possibilities.
- No meta talk about prompts, models, or pipelines.
- Do not restate the task as the answer; reason from visual evidence.
- The only edits allowed are pasting the candidate into the bottom-right cell and necessary size matching for that cell. All other pixels must remain identical to the reference.

Inputs:"""


# Maze prompt (aligned with original maze.py)
MAZE_PROMPT = """You are a precise maze solver.

SEMANTICS (for all mazes)
- Black squares: walls (impassable)
- White squares: path (walkable)
- Blue dot: start (the agent)
- Green rectangular frame: goal (reaching any white cell inside the green frame counts as success)
- Legal moves: up, down, left, right only. One cell per step; no diagonals, no jumps; never cross walls.

OUTPUT FORMAT (STRICT)
1) MULTI-IMAGE MODE — generate a SEQUENCE OF SEPARATE IMAGES, one per move:
   - Each output image must depict the maze state AFTER applying exactly one legal move.
   - Do NOT include the initial (pre-move) state.
   - Keep palette/layout/scale identical to the input; only the blue dot moves.
   - The number of returned images MUST equal the number of moves in the final answer (see step 2).
   - Absolutely FORBIDDEN: any collage/montage/spritesheet/grid/multi-panel/side-by-side/stacked images; no arrows, captions, or overlays; no GIFs/animations/video.

2) After all step images, emit EXACTLY ONE LINE containing ONLY the final move list as a JSON array of lowercase strings, wrapped as:
   <ANSWER_JSON>["right","down","left"]</ANSWER_JSON>


NO EXTRAS
- No tools, no OCR, no explanations, and no text other than the single <ANSWER_JSON>…</ANSWER_JSON> line.
- Do not restate the instructions or the condition.

REMINDERS
- Decide the full path first, then emit the image sequence (one image per move), then the single <ANSWER_JSON> line.
- One move per image; images must be separate files/parts, not stitched together in any way."""


# Sliding puzzle prompt (aligned with original sliding.py)
SLIDING_PROMPT = """You are a precise sliding puzzle solver.

SEMANTICS
- The puzzle is a 3×3 grid with 8 colored tiles and one empty space (shown as red).
- A "move" slides an adjacent tile INTO the empty space.
- Moves are named by the direction the COLORED TILE moves (not the empty space).
- Legal moves: up, down, left, right only. One tile per step.

OUTPUT FORMAT (STRICT)
1) MULTI-IMAGE MODE — generate a SEQUENCE OF SEPARATE IMAGES, one per move:
   - Each output image must depict the puzzle state AFTER applying exactly one move.
   - Do NOT include the initial (pre-move) state.
   - Keep tile colors identical; only positions change.
   - The number of returned images MUST equal the number of moves in the final answer.
   - Absolutely FORBIDDEN: any collage/montage/spritesheet/grid/multi-panel/side-by-side/stacked images.

2) After all step images, emit EXACTLY ONE LINE containing ONLY the final move list as a JSON array of lowercase strings, wrapped as:
   <ANSWER_JSON>["down","right","up"]</ANSWER_JSON>

NO EXTRAS
- No explanations beyond the single <ANSWER_JSON>…</ANSWER_JSON> line.
- Do not restate the instructions."""


def jigsaw_doc_to_text_visual_cot(doc: Dict, lmms_eval_specific_kwargs: Dict = None) -> str:
    """Visual CoT prompt for jigsaw task - aligned with original Uni-MMMU."""
    return JIGSAW_PROMPT


def maze_doc_to_text_visual_cot(doc: Dict, lmms_eval_specific_kwargs: Dict = None) -> str:
    """Visual CoT prompt for maze task - aligned with original Uni-MMMU."""
    return MAZE_PROMPT


def sliding_doc_to_text_visual_cot(doc: Dict, lmms_eval_specific_kwargs: Dict = None) -> str:
    """Visual CoT prompt for sliding puzzle - aligned with original Uni-MMMU."""
    return SLIDING_PROMPT


# ======================================================================
# VisualPuzzles
# ======================================================================


MULTI_CHOICE_DIRECT_PROMPT = "Answer the question with the option's letter from the given choices directly."
COT_PROMPT = "Solve the multiple-choice question and then answer with the option letter from the given choices. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options. Think step by step before answering."
PROMPTS = {"MULTI_CHOICE_DIRECT_PROMPT": MULTI_CHOICE_DIRECT_PROMPT, "COT_PROMPT": COT_PROMPT}


def VisualPuzzles_doc_to_visual(doc):
    image = doc["image"]
    # Handle HuggingFace datasets image format (dict with 'bytes' or already PIL Image)
    if isinstance(image, dict):
        # If it's a dict, convert to PIL Image
        if "bytes" in image:
            import io
            image = Image.open(io.BytesIO(image["bytes"]))
        elif "path" in image:
            image = Image.open(image["path"])
        else:
            # Try to use the dict directly as Image (some formats)
            import numpy as np
            image = Image.fromarray(np.array(image))
    return [image]


def VisualPuzzles_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = "Question: " + doc["question"].strip()
    options = doc["options"]
    if options is not None:
        question += "\nOptions:\n(A) " + options[0] + "\n(B) " + options[1] + "\n(C) " + options[2] + "\n(D) " + options[3]
    else:
        question += "\nOptions: Choose from (A) (B) (C) (D) in the image."
    question += "\n" + PROMPTS[lmms_eval_specific_kwargs["prompt"]]
    return question


def parse_response(response, all_choices, index2ans):
    """
    Return the last letter appearing after 'ANSWER:' in the input text.
    If there's no match, return None.
    """
    pattern = r"Answer:\s*\(([A-Za-z])\)"  # Answer: (A)
    matches = re.findall(pattern, response)
    if matches:
        for match in matches[::-1]:
            if match in all_choices or match.upper() in all_choices:
                return match
    pattern = r"(?<!Final )Answer:\s*([A-Za-z])"  # Answer: A
    matches = re.findall(pattern, response)
    if matches:
        for match in matches[::-1]:
            if match in all_choices or match.upper() in all_choices:
                return match
    pattern = r"Answer:\s*([A-Za-z])"  # Answer: A
    matches = re.findall(pattern, response)
    if matches:
        for match in matches[::-1]:
            if match in all_choices or match.upper() in all_choices:
                return match
    pattern = r"\s*\(([A-Za-z])\)"  # e.g., (A) (B) (C) (D)
    matches = re.findall(pattern, response)
    if matches:
        for match in matches[::-1]:
            if match in all_choices or match.upper() in all_choices:
                return match
    response = " " + response.strip()
    pattern = r"\s*([A-Za-z])\)"  # e.g., A) B) C) D)
    matches = re.findall(pattern, response)
    if matches:
        for match in matches[::-1]:
            if match in all_choices or match.upper() in all_choices:
                return match
    pattern = r"\s*\{([A-Za-z])\}"  # e.g., {A} {B} {C} {D}
    matches = re.findall(pattern, response)
    if matches:
        for match in matches[::-1]:
            if match in all_choices or match.upper() in all_choices:
                return match
    pattern = r"\s*\$([A-Za-z])\$"  # e.g., $A$, $B$, $C$, $D$
    matches = re.findall(pattern, response)
    if matches:
        for match in matches[::-1]:
            if match in all_choices or match.upper() in all_choices:
                return match
    pattern = r" ([A-Da-d])\."  # e.g., A. B. C. D.
    matches = re.findall(pattern, response)
    if matches:
        for match in matches[::-1]:
            if match in all_choices or match.upper() in all_choices:
                return match
    pattern = r" ([A-Da-d])"  # e.g., A B C D
    matches = re.findall(pattern, response)
    if matches and len(response) <= 5:
        for match in matches[::-1]:
            if match in all_choices or match.upper() in all_choices:
                return match
    if index2ans is not None:
        for index in all_choices:
            ans = index2ans[index]
            if f"answer: {ans}" in response.lower():
                return index
            if f"answer:{ans}" in response.lower():
                return index
        last_found = None
        last_index = -1
        for index in all_choices:
            ans = index2ans[index]
            idx = response.rfind(ans)
            if idx > last_index:
                last_found = index
                last_index = idx
        if last_found:
            return last_found
    return None


def VisualPuzzles_process_result(doc, results):
    """
    Process results with category-based metrics

    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (category), value: metric dict
    """
    pred = results[0].strip()
    all_choices = ["A", "B", "C", "D"]
    if doc["options"] == None:
        index2ans = None
    else:
        index2ans = {all_choices[i]: doc["options"][i] for i in range(4)}
    pred = parse_response(pred, all_choices, index2ans)
    target = doc["answer"]

    # Calculate score
    score = 1.0 if pred.lower() == target.lower() else 0.0

    # Get category from doc
    category = doc.get("category", "Unknown")

    # Return results for both category-specific and overall metrics
    result_dict = {
        category: {
            "question_id": doc.get("id", doc.get("idx", "unknown")),
            "category": category,
            "score": score
        },
        "average": {
            "question_id": doc.get("id", doc.get("idx", "unknown")),
            "category": category,
            "score": score
        }
    }

    return result_dict


def VisualPuzzles_aggregate_results(results):
    """
    Aggregate results by category

    Args:
        results: a list of values returned by process_results
    Returns:
        Average score for the category
    """
    category_scores = defaultdict(list)

    for result in results:
        score = result["score"]
        category = result["category"]
        category_scores[category].append(score)

    # Calculate average for each category
    category_avg_scores = {}
    for category, scores in category_scores.items():
        avg_score = sum(scores) / len(scores) if scores else 0.0
        category_avg_scores[category] = avg_score
        eval_logger.info(f"{category}: {avg_score:.4f} ({len(scores)} samples)")

    # Calculate overall average
    all_scores = [score for scores in category_scores.values() for score in scores]
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

    return avg_score


def VisualPuzzles_process_result_simple(doc, results):
    """
    Simplified process results for single-category tasks.

    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with accuracy score
    """
    pred = results[0].strip()
    all_choices = ["A", "B", "C", "D"]
    if doc["options"] == None:
        index2ans = None
    else:
        index2ans = {all_choices[i]: doc["options"][i] for i in range(4)}
    pred = parse_response(pred, all_choices, index2ans)
    target = doc["answer"]

    # Calculate score
    score = 1.0 if pred.lower() == target.lower() else 0.0

    return {"accuracy": score}


def VisualPuzzles_aggregate_simple(results):
    """
    Simple aggregation for single-category tasks.

    Args:
        results: a list of score values
    Returns:
        Average accuracy
    """
    if not results:
        return 0.0
    return sum(results) / len(results)


# ============================================================
# Visual CoT Prompt Functions (Category-Specific)
# ============================================================

def VisualPuzzles_doc_to_text_visual_cot_algorithmic(doc, lmms_eval_specific_kwargs=None):
    """
    Visual CoT prompt for ALGORITHMIC reasoning puzzles.
    Stage 1: Identify numerical/symbolic patterns and computation rules.
    """
    question = "Question: " + doc["question"].strip()
    options = doc["options"]
    if options is not None:
        question += "\nOptions:\n(A) " + options[0] + "\n(B) " + options[1] + "\n(C) " + options[2] + "\n(D) " + options[3]
    else:
        question += "\nOptions: Choose from (A) (B) (C) (D) in the image."

    # Stage 1: Generate auxiliary visualization for algorithmic reasoning
    generation_prompt = f"""You are given an algorithmic reasoning puzzle. Analyze the puzzle and create a helpful visualization.

{question}

Your task:
1. Identify any numerical sequences, patterns, or computational rules in the puzzle
2. Create a diagram that clearly shows:
   - The step-by-step computation or transformation process
   - Arrows or annotations showing how numbers/symbols change
   - The mathematical relationship or formula discovered
   - Highlighted patterns (e.g., +2, ×3, alternating, etc.)
3. Label each step of the algorithm clearly

Generate a clear diagram that reveals the underlying algorithmic pattern."""

    # Stage 2: Solve using both images
    question_prompt = f"""{question}

You are given TWO images:
1) ORIGINAL PUZZLE: The algorithmic reasoning puzzle
2) AUXILIARY DIAGRAM: A visualization showing the computational pattern and steps

Use the auxiliary diagram to understand the algorithm, then select the correct answer.
Answer with the option letter (A, B, C, or D) directly."""

    return f"[GEN_PROMPT]{generation_prompt}[/GEN_PROMPT]\n[QUESTION]{question_prompt}[/QUESTION]"


def VisualPuzzles_doc_to_text_visual_cot_analogical(doc, lmms_eval_specific_kwargs=None):
    """
    Visual CoT prompt for ANALOGICAL reasoning puzzles.
    Stage 1: Identify transformation relationships between elements.
    """
    question = "Question: " + doc["question"].strip()
    options = doc["options"]
    if options is not None:
        question += "\nOptions:\n(A) " + options[0] + "\n(B) " + options[1] + "\n(C) " + options[2] + "\n(D) " + options[3]
    else:
        question += "\nOptions: Choose from (A) (B) (C) (D) in the image."

    # Stage 1: Generate auxiliary visualization for analogical reasoning
    generation_prompt = f"""You are given an analogical reasoning puzzle (A is to B as C is to ?).

{question}

Your task:
1. Identify the transformation relationship between the first pair of elements
2. Create a diagram that clearly shows:
   - What changes occur (rotation, reflection, color change, size change, addition/removal of elements)
   - Arrows indicating the direction and type of transformation
   - Labels describing each transformation (e.g., "rotate 90°", "invert colors", "add dot")
   - Apply the same transformation to show what the answer should look like
3. Make the analogy relationship visually explicit

Generate a diagram that reveals the transformation pattern between pairs."""

    # Stage 2: Solve using both images
    question_prompt = f"""{question}

You are given TWO images:
1) ORIGINAL PUZZLE: The analogical reasoning puzzle
2) AUXILIARY DIAGRAM: A visualization showing the transformation relationship

Use the auxiliary diagram to understand how A transforms to B, apply the same rule to C, then select the correct answer.
Answer with the option letter (A, B, C, or D) directly."""

    return f"[GEN_PROMPT]{generation_prompt}[/GEN_PROMPT]\n[QUESTION]{question_prompt}[/QUESTION]"


def VisualPuzzles_doc_to_text_visual_cot_deductive(doc, lmms_eval_specific_kwargs=None):
    """
    Visual CoT prompt for DEDUCTIVE reasoning puzzles.
    Stage 1: Map out logical rules and inference chains.
    """
    question = "Question: " + doc["question"].strip()
    options = doc["options"]
    if options is not None:
        question += "\nOptions:\n(A) " + options[0] + "\n(B) " + options[1] + "\n(C) " + options[2] + "\n(D) " + options[3]
    else:
        question += "\nOptions: Choose from (A) (B) (C) (D) in the image."

    # Stage 1: Generate auxiliary visualization for deductive reasoning
    generation_prompt = f"""You are given a deductive reasoning puzzle that requires logical inference.

{question}

Your task:
1. Identify the given premises, rules, or constraints in the puzzle
2. Create a diagram that clearly shows:
   - All given conditions/rules listed clearly
   - A logical flowchart or inference chain
   - Step-by-step deduction from premises to conclusion
   - Elimination of incorrect possibilities
   - The logical path leading to the answer
3. Use arrows to show the deduction flow

Generate a logical inference diagram that traces the reasoning path."""

    # Stage 2: Solve using both images
    question_prompt = f"""{question}

You are given TWO images:
1) ORIGINAL PUZZLE: The deductive reasoning puzzle
2) AUXILIARY DIAGRAM: A logical inference diagram showing the deduction steps

Follow the logical chain in the auxiliary diagram to reach the conclusion, then select the correct answer.
Answer with the option letter (A, B, C, or D) directly."""

    return f"[GEN_PROMPT]{generation_prompt}[/GEN_PROMPT]\n[QUESTION]{question_prompt}[/QUESTION]"


def VisualPuzzles_doc_to_text_visual_cot_inductive(doc, lmms_eval_specific_kwargs=None):
    """
    Visual CoT prompt for INDUCTIVE reasoning puzzles.
    Stage 1: Identify repeating patterns and generalize rules from examples.
    """
    question = "Question: " + doc["question"].strip()
    options = doc["options"]
    if options is not None:
        question += "\nOptions:\n(A) " + options[0] + "\n(B) " + options[1] + "\n(C) " + options[2] + "\n(D) " + options[3]
    else:
        question += "\nOptions: Choose from (A) (B) (C) (D) in the image."

    # Stage 1: Generate auxiliary visualization for inductive reasoning
    generation_prompt = f"""You are given an inductive reasoning puzzle that requires pattern recognition.

{question}

Your task:
1. Observe the sequence of examples and identify the underlying pattern
2. Create a diagram that clearly shows:
   - The repeating elements or motifs highlighted/circled
   - The progression rule (what changes from one step to the next)
   - Annotations showing the pattern cycle or growth rule
   - A prediction of what comes next based on the pattern
   - Color-coding or numbering to show pattern repetition
3. Make the inductive pattern visually obvious

Generate a diagram that highlights the repeating pattern and predicts the next element."""

    # Stage 2: Solve using both images
    question_prompt = f"""{question}

You are given TWO images:
1) ORIGINAL PUZZLE: The inductive reasoning puzzle
2) AUXILIARY DIAGRAM: A visualization highlighting the pattern and its progression

Use the auxiliary diagram to understand the pattern rule, then select the answer that continues the pattern correctly.
Answer with the option letter (A, B, C, or D) directly."""

    return f"[GEN_PROMPT]{generation_prompt}[/GEN_PROMPT]\n[QUESTION]{question_prompt}[/QUESTION]"


def VisualPuzzles_doc_to_text_visual_cot_spatial(doc, lmms_eval_specific_kwargs=None):
    """
    Visual CoT prompt for SPATIAL reasoning puzzles.
    Stage 1: Visualize rotations, folding, or 3D transformations.
    """
    question = "Question: " + doc["question"].strip()
    options = doc["options"]
    if options is not None:
        question += "\nOptions:\n(A) " + options[0] + "\n(B) " + options[1] + "\n(C) " + options[2] + "\n(D) " + options[3]
    else:
        question += "\nOptions: Choose from (A) (B) (C) (D) in the image."

    # Stage 1: Generate auxiliary visualization for spatial reasoning
    generation_prompt = f"""You are given a spatial reasoning puzzle involving 3D visualization or transformations.

{question}

Your task:
1. Analyze the spatial transformation required (rotation, folding, unfolding, different viewpoint)
2. Create a diagram that clearly shows:
   - The object from multiple angles if rotation is involved
   - Step-by-step folding/unfolding process if applicable
   - Arrows indicating rotation direction and degree
   - Reference points or markers to track orientation
   - The resulting shape after transformation
3. Add axis lines or reference frames to clarify spatial orientation

Generate a multi-view or step-by-step transformation diagram."""

    # Stage 2: Solve using both images
    question_prompt = f"""{question}

You are given TWO images:
1) ORIGINAL PUZZLE: The spatial reasoning puzzle
2) AUXILIARY DIAGRAM: A visualization showing the spatial transformation from multiple views

Use the auxiliary diagram to mentally trace the transformation, then select the correct answer.
Answer with the option letter (A, B, C, or D) directly."""

    return f"[GEN_PROMPT]{generation_prompt}[/GEN_PROMPT]\n[QUESTION]{question_prompt}[/QUESTION]"

