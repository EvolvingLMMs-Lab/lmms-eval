"""VSP (Visual Spatial Planning) task utilities for lmms-eval.

Tasks: Google Map Navigation, Collision Detection
Paper: https://arxiv.org/abs/2407.01863
"""

import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import snapshot_download
from loguru import logger as eval_logger
from PIL import Image

with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
    config = yaml.safe_load("".join(safe_data))

DATASET_PATH = config.get("dataset_path", "AnonyCAD/VSP")

try:
    cache_dir = snapshot_download(
        repo_id=DATASET_PATH,
        repo_type="dataset",
        local_dir_use_symlinks=False,
    )
except Exception as e:
    eval_logger.warning(f"Could not download VSP dataset: {e}")
    cache_dir = os.getenv("VSP_DATA_PATH", "")


GMAP_PROMPT = """As a professional pathfinder, your task is to analyze a map and find a route from the starting location to the goal. Since coding is not within your skill set, your approach relies on logical reasoning of the map.

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

[Test Image]"""


COLLISION_PROMPT_TEMPLATE = """As a professional navigation agent, your task is to analyze a map and determine the time needed for the car and the person passing the goal.

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

The car is moving {car_dir} with speed {car_speed}, and the person is moving {person_dir} with speed {person_speed}."""


def _load_image(img_data: Any) -> Image.Image | None:
    """Load image from various data formats."""
    if img_data is None:
        return None
    if isinstance(img_data, bytes):
        return Image.open(BytesIO(img_data)).convert("RGB")
    if isinstance(img_data, Image.Image):
        return img_data.convert("RGB")
    if isinstance(img_data, str) and os.path.exists(img_data):
        return Image.open(img_data).convert("RGB")
    if isinstance(img_data, str) and cache_dir:
        full_path = os.path.join(cache_dir, img_data)
        if os.path.exists(full_path):
            return Image.open(full_path).convert("RGB")
    return None


def vsp_doc_to_visual(doc: dict[str, Any]) -> list[Image.Image]:
    """Generic visual extractor for VSP tasks."""
    images = []
    image_keys = ["image", "example_image", "test_image", "icon_image"]
    for key in image_keys:
        if key in doc:
            img = _load_image(doc[key])
            if img is not None:
                images.append(img)
    return images


def vsp_doc_to_text(
    doc: dict[str, Any], lmms_eval_specific_kwargs: dict[str, Any] | None = None
) -> str:
    """Generic text extractor for VSP tasks."""
    return doc.get("question", doc.get("prompt", ""))


def gmap_doc_to_visual(doc: dict[str, Any]) -> list[Image.Image]:
    """Get visual inputs for google map task."""
    images = []
    for key in ["example_image", "test_image"]:
        if key in doc and doc[key]:
            img = _load_image(doc[key])
            if img is not None:
                images.append(img)
    if not images and "image" in doc:
        img = _load_image(doc["image"])
        if img is not None:
            images.append(img)
    return images


def gmap_doc_to_text(
    doc: dict[str, Any], lmms_eval_specific_kwargs: dict[str, Any] | None = None
) -> str:
    """Get prompt for google map task."""
    if "prompt" in doc:
        return doc["prompt"]
    return GMAP_PROMPT


def gmap_process_results(
    doc: dict[str, Any], results: list[str]
) -> dict[str, dict[str, Any]]:
    """Process google map results - parse directions and compare."""
    result_text = results[0] if results else ""

    north_south = 0
    east_west = 0

    try:
        c_index = result_text.find("Output")
        if c_index == -1:
            c_index = result_text.find("Path")
        if c_index != -1:
            contents = result_text[c_index + 7 :]
        else:
            contents = result_text

        contents = contents.replace('"', "").replace("'", "").replace(".", "")
        lines = contents.strip().split("\n")

        for line in lines:
            line_lower = line.lower()
            if "north" in line_lower:
                match = re.search(r"north[:\s]*(\d+)", line_lower)
                if match:
                    north_south += int(match.group(1))
            if "south" in line_lower:
                match = re.search(r"south[:\s]*(\d+)", line_lower)
                if match:
                    north_south -= int(match.group(1))
            if "east" in line_lower:
                match = re.search(r"east[:\s]*(\d+)", line_lower)
                if match:
                    east_west += int(match.group(1))
            if "west" in line_lower:
                match = re.search(r"west[:\s]*(\d+)", line_lower)
                if match:
                    east_west -= int(match.group(1))
    except Exception as e:
        eval_logger.error(f"Error parsing google map result: {e}")

    gt_ns = doc.get("gt_north_south", 0)
    gt_ew = doc.get("gt_east_west", 0)

    correct = 1.0 if (north_south == gt_ns and east_west == gt_ew) else 0.0

    return {
        "gmap_acc": {
            "test_id": doc.get("test_id", doc.get("id", "")),
            "score": correct,
            "pred_ns": north_south,
            "pred_ew": east_west,
            "gt_ns": gt_ns,
            "gt_ew": gt_ew,
        }
    }


def gmap_aggregate_results(results: list[dict[str, Any]]) -> float:
    """Aggregate google map results."""
    scores = [r["score"] for r in results]
    acc = sum(scores) / len(scores) if scores else 0.0
    eval_logger.info(
        f"Google Map Accuracy: {acc:.4f} ({sum(scores):.0f}/{len(scores)})"
    )
    return acc


def collision_doc_to_visual(doc: dict[str, Any]) -> list[Image.Image]:
    """Get visual inputs for collision task."""
    images = []
    for key in ["icon_image", "example_image", "test_image"]:
        if key in doc and doc[key]:
            img = _load_image(doc[key])
            if img is not None:
                images.append(img)
    if not images and "image" in doc:
        img = _load_image(doc["image"])
        if img is not None:
            images.append(img)
    return images


def collision_doc_to_text(
    doc: dict[str, Any], lmms_eval_specific_kwargs: dict[str, Any] | None = None
) -> str:
    """Get prompt for collision task."""
    if "prompt" in doc:
        return doc["prompt"]
    return COLLISION_PROMPT_TEMPLATE.format(
        car_dir=doc.get("car_dir", ""),
        car_speed=doc.get("car_speed", ""),
        person_dir=doc.get("person_dir", ""),
        person_speed=doc.get("person_speed", ""),
    )


def collision_process_results(
    doc: dict[str, Any], results: list[str]
) -> dict[str, dict[str, Any]]:
    """Process collision results - parse Car/Person times and compare."""
    result_text = results[0] if results else ""

    pattern = r"(Car|Person):\s*([0-9]*\.?[0-9]+)"
    matches = re.findall(pattern, result_text, re.IGNORECASE)
    info = {match[0].capitalize(): float(match[1]) for match in matches}

    gt_car = doc.get("gt_car", 0.0)
    gt_person = doc.get("gt_person", 0.0)

    pred_car = info.get("Car", -999)
    pred_person = info.get("Person", -999)

    car_correct = 1.0 if abs(pred_car - gt_car) <= 1.0 else 0.0
    person_correct = 1.0 if abs(pred_person - gt_person) <= 1.0 else 0.0

    return {
        "collision_acc": {
            "test_id": doc.get("test_id", doc.get("id", "")),
            "car_score": car_correct,
            "person_score": person_correct,
            "score": (car_correct + person_correct) / 2,
            "pred_car": pred_car,
            "pred_person": pred_person,
            "gt_car": gt_car,
            "gt_person": gt_person,
        }
    }


def collision_aggregate_results(results: list[dict[str, Any]]) -> float:
    """Aggregate collision results."""
    car_scores = [r["car_score"] for r in results]
    person_scores = [r["person_score"] for r in results]

    car_acc = sum(car_scores) / len(car_scores) if car_scores else 0.0
    person_acc = sum(person_scores) / len(person_scores) if person_scores else 0.0
    overall_acc = (car_acc + person_acc) / 2

    eval_logger.info(
        f"  Car Accuracy: {car_acc:.4f} ({sum(car_scores):.0f}/{len(car_scores)})"
    )
    eval_logger.info(
        f"  Person Accuracy: {person_acc:.4f} "
        f"({sum(person_scores):.0f}/{len(person_scores)})"
    )
    eval_logger.info(f"  Overall Accuracy: {overall_acc:.4f}")

    return overall_acc
