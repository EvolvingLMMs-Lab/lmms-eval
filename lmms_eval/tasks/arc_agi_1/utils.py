import json
import re
from typing import Any

from loguru import logger as eval_logger
from PIL import Image

CODE_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
GRID_PATTERN = re.compile(r"\[\s*\[.*?\]\s*\]", re.DOTALL)


def _extract_first_image(image_value):
    if isinstance(image_value, (list, tuple)):
        if len(image_value) == 0:
            return None
        image_value = image_value[0]

    if isinstance(image_value, Image.Image):
        return image_value.convert("RGB")

    if hasattr(image_value, "convert"):
        try:
            return image_value.convert("RGB")
        except Exception as err:
            eval_logger.warning("Failed to convert ARC-AGI image to RGB: {}", err)

    return None


def _extract_first_value(value: Any, default: Any = None) -> Any:
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return default
        value = value[0]
    if value is None:
        return default
    return value


def _is_grid_of_ints(value):
    if not isinstance(value, list):
        return False
    for row in value:
        if not isinstance(row, list):
            return False
        for cell in row:
            if isinstance(cell, bool) or not isinstance(cell, int):
                return False
    return True


def _try_parse_grid(candidate):
    if not isinstance(candidate, str):
        return None

    candidate = candidate.strip()
    if not candidate:
        return None

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    if _is_grid_of_ints(parsed):
        return parsed
    return None


def _get_raw_solution_grid(doc):
    raw_solution = doc.get("raw_solution")
    target = _extract_first_value(raw_solution, default=[])
    if _is_grid_of_ints(target):
        return target
    eval_logger.warning("Invalid ARC-AGI raw_solution format for id={}", doc.get("id", "unknown"))
    return None


def arc_agi_1_doc_to_visual(doc):
    stacked_train_image = _extract_first_image(doc.get("stacked_train_image"))
    test_input_image = _extract_first_image(doc.get("test_images"))

    if stacked_train_image is None:
        eval_logger.warning("Missing stacked_train_image for id={}", doc.get("id", "unknown"))
    if test_input_image is None:
        eval_logger.warning("Missing test_images for id={}", doc.get("id", "unknown"))

    if stacked_train_image is None or test_input_image is None:
        raise ValueError(f"ARC-AGI-1 visual inputs are incomplete for id={doc.get('id', 'unknown')}")

    return [stacked_train_image, test_input_image]


def arc_agi_1_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    test_input_text = _extract_first_value(doc.get("test_inputs"), default="")

    prompt = (
        "You are given demonstration input-output pairs as images. "
        "Based on the pattern, predict the output grid for the test input.\n"
        "The first image contains all demonstration input/output pairs.\n"
        "The second image contains the test input grid.\n\n"
        f"Text representation of the test input grid:\n{test_input_text}\n\n"
        "Output only the predicted output grid as a JSON array of arrays of integers.\n"
        "Example: [[1,2],[3,4]]"
    )

    return f"{pre_prompt}{prompt}{post_prompt}"


def arc_agi_1_doc_to_target(doc):
    target = _extract_first_value(doc.get("raw_solution"), default=[])
    return json.dumps(target)


def _parse_grid_from_response(text: str) -> list[list[int]] | None:
    if not isinstance(text, str):
        return None

    text = text.strip()
    if not text:
        return None

    parsed_grid = _try_parse_grid(text)
    if parsed_grid is not None:
        return parsed_grid

    for block in CODE_FENCE_PATTERN.findall(text):
        parsed_grid = _try_parse_grid(block)
        if parsed_grid is not None:
            return parsed_grid

    for match in GRID_PATTERN.findall(text):
        parsed_grid = _try_parse_grid(match)
        if parsed_grid is not None:
            return parsed_grid

    return None


def arc_agi_1_process_results(doc, results):
    response_text = results[0] if results else ""
    if not isinstance(response_text, str):
        response_text = str(response_text)

    parsed_grid = _parse_grid_from_response(response_text)
    target_grid = _get_raw_solution_grid(doc)

    if parsed_grid is None:
        eval_logger.debug("Failed to parse model output as ARC grid for id={}", doc.get("id", "unknown"))
        return {"arc_agi_1_acc": 0.0}

    if target_grid is None:
        return {"arc_agi_1_acc": 0.0}

    return {"arc_agi_1_acc": 1.0 if parsed_grid == target_grid else 0.0}
