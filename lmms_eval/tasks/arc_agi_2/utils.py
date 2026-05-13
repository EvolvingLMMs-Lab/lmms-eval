import json
import re
from typing import Any

from loguru import logger as eval_logger
from PIL import Image

_CODE_FENCE_PATTERN = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_DOUBLE_BRACKET_PATTERN = re.compile(r"\[\s*\[.*?\]\s*\]", re.DOTALL)


def _to_rgb_image(image_obj: Any) -> Image.Image | None:
    if isinstance(image_obj, Image.Image):
        return image_obj.convert("RGB")
    if hasattr(image_obj, "convert"):
        try:
            return image_obj.convert("RGB")
        except Exception:
            return None
    return None


def arc_agi_2_doc_to_visual(doc: dict) -> list[Image.Image]:
    visuals = []
    train_inputs = doc.get("train_input_image_color") or []
    train_outputs = doc.get("train_output_image_color") or []

    if len(train_inputs) != len(train_outputs):
        eval_logger.warning(
            "ARC-AGI-2 train pair mismatch for task {}: {} inputs vs {} outputs",
            doc.get("id", doc.get("task_id", "unknown")),
            len(train_inputs),
            len(train_outputs),
        )

    pair_count = min(len(train_inputs), len(train_outputs))
    for idx in range(pair_count):
        train_input = _to_rgb_image(train_inputs[idx])
        train_output = _to_rgb_image(train_outputs[idx])

        if train_input is not None:
            visuals.append(train_input)
        else:
            eval_logger.warning(
                "ARC-AGI-2 missing/invalid train input image at pair {} for task {}",
                idx,
                doc.get("id", doc.get("task_id", "unknown")),
            )

        if train_output is not None:
            visuals.append(train_output)
        else:
            eval_logger.warning(
                "ARC-AGI-2 missing/invalid train output image at pair {} for task {}",
                idx,
                doc.get("id", doc.get("task_id", "unknown")),
            )

    test_inputs = doc.get("test_input_image_color") or []
    if test_inputs:
        test_image = _to_rgb_image(test_inputs[0])
        if test_image is not None:
            visuals.append(test_image)
        else:
            eval_logger.warning(
                "ARC-AGI-2 missing/invalid test input image for task {}",
                doc.get("id", doc.get("task_id", "unknown")),
            )
    else:
        eval_logger.warning(
            "ARC-AGI-2 has no test input image for task {}",
            doc.get("id", doc.get("task_id", "unknown")),
        )

    return visuals


def arc_agi_2_doc_to_text(doc: dict, lmms_eval_specific_kwargs=None) -> str:
    lmms_eval_specific_kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    train_inputs = doc.get("train_input_image_color") or []
    train_outputs = doc.get("train_output_image_color") or []
    pair_count = min(len(train_inputs), len(train_outputs))

    image_refs = []
    for idx in range(pair_count):
        image_refs.append(f"Image {2 * idx + 1} is demo input {idx + 1}.")
        image_refs.append(f"Image {2 * idx + 2} is demo output {idx + 1}.")
    image_refs.append(f"Image {2 * pair_count + 1} is the final test input.")

    test_input_texts = doc.get("test_input_texts") or [""]
    test_input_text = test_input_texts[0] if test_input_texts else ""

    prompt = (
        "You are solving an ARC visual reasoning task. "
        "Each task contains demo input/output grid pairs where the output is generated from the input by a hidden rule.\n\n"
        "Image references:\n"
        f"{' '.join(image_refs)}\n\n"
        "Based on the pattern in the demo pairs, predict the output grid for the final test input image.\n"
        "The test input is also provided in text grid form for clarity:\n"
        f"{test_input_text}\n\n"
        "Return the final predicted grid as a JSON array of arrays, for example [[1,2],[3,4]].\n"
        "Return only the JSON array with no additional text."
    )

    parts = []
    if pre_prompt:
        parts.append(pre_prompt)
    parts.append(prompt)
    if post_prompt:
        parts.append(post_prompt)
    return "\n\n".join(parts)


def arc_agi_2_doc_to_target(doc: dict) -> str:
    test_targets = doc.get("test_targets") or []
    if not test_targets:
        return "[]"
    return test_targets[0]


def _try_parse_grid_candidate(candidate: str) -> list[list[int]] | None:
    candidate = candidate.strip()
    if not candidate:
        return None

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    if isinstance(parsed, dict) and isinstance(parsed.get("text"), str):
        return _parse_grid_from_response(parsed["text"])

    return _normalize_grid(parsed)


def _extract_bracket_candidates(text: str) -> list[str]:
    candidates = []
    depth = 0
    start = None
    in_string = False
    escaped = False

    for idx, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char == "[":
            if depth == 0:
                start = idx
            depth += 1
            continue

        if char == "]" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start : idx + 1])
                start = None

    return candidates


def _parse_grid_from_response(text: str) -> list[list[int]] | None:
    if text is None:
        return None

    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    if not text:
        return None

    parsed = _try_parse_grid_candidate(text)
    if parsed is not None:
        return parsed

    for fenced_content in _CODE_FENCE_PATTERN.findall(text):
        parsed = _try_parse_grid_candidate(fenced_content)
        if parsed is not None:
            return parsed

    for bracket_match in _DOUBLE_BRACKET_PATTERN.findall(text):
        parsed = _try_parse_grid_candidate(bracket_match)
        if parsed is not None:
            return parsed

    for candidate in _extract_bracket_candidates(text):
        parsed = _try_parse_grid_candidate(candidate)
        if parsed is not None:
            return parsed

    return None


def _normalize_grid(grid: Any) -> list[list[int]] | None:
    if isinstance(grid, str):
        try:
            grid = json.loads(grid)
        except json.JSONDecodeError:
            return None

    if not isinstance(grid, list):
        return None

    normalized_grid = []
    for row in grid:
        if not isinstance(row, list):
            return None

        normalized_row = []
        for value in row:
            if isinstance(value, bool):
                return None
            if isinstance(value, int):
                normalized_row.append(value)
                continue
            if isinstance(value, float) and value.is_integer():
                normalized_row.append(int(value))
                continue
            if isinstance(value, str):
                stripped = value.strip()
                if re.fullmatch(r"-?\d+", stripped) is None:
                    return None
                normalized_row.append(int(stripped))
                continue
            return None
        normalized_grid.append(normalized_row)

    return normalized_grid


def arc_agi_2_process_results(doc: dict, results: list[str]) -> dict[str, float]:
    response_text = results[0] if results else ""
    predicted_grid = _parse_grid_from_response(response_text)

    raw_test_outputs = doc.get("test_outputs") or []
    ground_truth_raw = raw_test_outputs[0] if raw_test_outputs else None
    ground_truth_grid = _normalize_grid(ground_truth_raw)
    if ground_truth_grid is None and isinstance(ground_truth_raw, str):
        ground_truth_grid = _parse_grid_from_response(ground_truth_raw)

    if predicted_grid is None:
        eval_logger.debug(
            "ARC-AGI-2 failed to parse prediction for task {}",
            doc.get("id", doc.get("task_id", "unknown")),
        )
    if ground_truth_grid is None:
        eval_logger.warning(
            "ARC-AGI-2 failed to parse ground truth for task {}",
            doc.get("id", doc.get("task_id", "unknown")),
        )

    score = 1.0 if predicted_grid is not None and predicted_grid == ground_truth_grid else 0.0
    return {"arc_agi_2_acc": score}
