"""Exact function-level export of the internal evaluation implementation."""

from __future__ import annotations

import json
import math
import re
from typing import Any, Mapping, Sequence

ANSWER_LETTERS = tuple("ABCDE")


def format_number(value: Any) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite number: {value!r}")
    text = f"{number:.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def parse_prediction(task: str, raw_output: Any) -> tuple[Any | None, str | None]:
    if task == "TG":
        return parse_tg_prediction(raw_output)
    if task == "SG":
        return parse_sg_prediction(raw_output)
    if task == "QA":
        return parse_qa_prediction(raw_output)
    return None, f"unsupported_task_{task}"


def parse_tg_prediction(raw_output: Any) -> tuple[list[float] | None, str | None]:
    if isinstance(raw_output, str):
        try:
            payload = json.loads(raw_output.strip())
        except json.JSONDecodeError:
            return None, "TG_output_must_be_a_JSON_object"
    else:
        payload = raw_output
    expected = {"start_time", "end_time"}
    if not isinstance(payload, Mapping) or set(payload.keys()) != expected:
        return None, "TG_output_must_contain_exactly_start_time_and_end_time"
    try:
        start = float(payload["start_time"])
        end = float(payload["end_time"])
    except (TypeError, ValueError):
        return None, "TG_timestamps_must_be_numeric"
    if not all(math.isfinite(value) for value in (start, end)) or not 0 <= start <= end:
        return None, "TG_interval_is_invalid"
    return [start, end], None


def parse_sg_prediction(raw_output: Any) -> tuple[dict[str, Any] | None, str | None]:
    if isinstance(raw_output, str):
        try:
            payload = json.loads(raw_output.strip())
        except json.JSONDecodeError:
            return None, "SG_output_must_be_a_JSON_object"
    else:
        payload = raw_output
    expected = {"start_time", "start_bbox", "end_time", "end_bbox"}
    if not isinstance(payload, Mapping) or set(payload.keys()) != expected:
        return None, "SG_output_must_contain_exactly_four_fields"
    try:
        start = float(payload["start_time"])
        end = float(payload["end_time"])
        start_box = [float(value) for value in payload["start_bbox"]]
        end_box = [float(value) for value in payload["end_bbox"]]
    except (TypeError, ValueError):
        return None, "SG_timestamps_and_boxes_must_be_numeric"
    if not (math.isfinite(start) and math.isfinite(end) and 0 <= start <= end):
        return None, "SG_interval_is_invalid"
    for box in (start_box, end_box):
        if len(box) != 4 or not all(math.isfinite(value) for value in box):
            return None, "SG_bbox_must_have_four_finite_coordinates"
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
            return None, "SG_bbox_must_be_valid_norm1000_xyxy"
    return {
        "start_time": start,
        "start_bbox": start_box,
        "end_time": end,
        "end_bbox": end_box,
    }, None


def parse_qa_prediction(raw_output: Any) -> tuple[str | None, str | None]:
    if not isinstance(raw_output, str):
        return None, "QA_output_must_be_a_string"
    answer = raw_output.strip().upper()
    if answer not in ANSWER_LETTERS:
        return None, "QA_output_must_be_one_letter_A_to_E"
    return answer, None


def _extract_json_object(raw_output: Any) -> dict[str, Any] | None:
    if isinstance(raw_output, Mapping):
        return dict(raw_output)
    if not isinstance(raw_output, str):
        return None
    stripped = raw_output.strip()
    fenced = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        stripped,
        re.DOTALL | re.IGNORECASE,
    )
    if fenced:
        stripped = fenced.group(1).strip()
    decoder = json.JSONDecoder()
    for index, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        return payload if isinstance(payload, dict) else None
    return None


def _extract_numbers(text: str) -> list[float]:
    return [float(value) for value in re.findall(r"(?<![A-Za-z])-?\d+(?:\.\d+)?", text)]


_TG_GAP = r"(?:\s|[*_`~\"'])*"

_TG_NUMBER = r"(\d+(?:\.\d+)?)"

_TG_UNIT = r"(?:\s*(?:s|sec|secs|second|seconds)\b)?"

_TG_CLOCK = r"(\d{1,2}:\d{2}(?::\d{2})?)"

_TG_BRACKET = re.compile(rf"\[\s*{_TG_NUMBER}{_TG_UNIT}{_TG_GAP},\s*{_TG_NUMBER}{_TG_UNIT}\s*\]")

_TG_CLOCK_BRACKET = re.compile(rf"\[\s*{_TG_CLOCK}{_TG_GAP},\s*{_TG_CLOCK}\s*\]")

_TG_LABELED_CLOCK = re.compile(
    rf"start(?:\s*_?\s*(?:time|sec|second))?[\"']?{_TG_GAP}"
    rf"(?:of|is|at|:|=)?{_TG_GAP}{_TG_CLOCK}.{{0,80}}?"
    rf"end(?:\s*_?\s*(?:time|sec|second))?[\"']?{_TG_GAP}"
    rf"(?:of|is|at|:|=)?{_TG_GAP}{_TG_CLOCK}",
    re.IGNORECASE | re.DOTALL,
)

_TG_LABELED = re.compile(
    rf"start(?:\s*_?\s*(?:time|sec|second))?{_TG_GAP}"
    rf"(?:of|is|at|:|=)?{_TG_GAP}{_TG_NUMBER}{_TG_UNIT}.{{0,80}}?"
    rf"end(?:\s*_?\s*(?:time|sec|second))?{_TG_GAP}"
    rf"(?:of|is|at|:|=)?{_TG_GAP}{_TG_NUMBER}",
    re.IGNORECASE | re.DOTALL,
)

_TG_PROSE = re.compile(
    rf"(?:from|between|start(?:ing|s)?(?:\s+at)?|at|"
    rf"occurs?(?:\s+(?:from|between))?){_TG_GAP}{_TG_NUMBER}{_TG_UNIT}"
    rf"{_TG_GAP}(?:to|until|till|and|through|–|—|-){_TG_GAP}{_TG_NUMBER}",
    re.IGNORECASE,
)

_TG_PROSE_CLOCK = re.compile(
    rf"(?:from|between|start(?:ing|s)?(?:\s+at)?|at){_TG_GAP}{_TG_CLOCK}"
    rf"{_TG_GAP}(?:to|until|till|and|through|–|—|-){_TG_GAP}{_TG_CLOCK}",
    re.IGNORECASE,
)


def _clock_text_to_seconds(value: str) -> float:
    parts = [int(part) for part in value.split(":")]
    if len(parts) == 2:
        return float(parts[0] * 60 + parts[1])
    return float(parts[0] * 3600 + parts[1] * 60 + parts[2])


def parse_tg_prediction_for_scoring(
    raw_output: Any,
) -> tuple[list[float] | None, str | None]:
    """Parse an explicitly stated interval while retaining notation provenance."""

    payload = _extract_json_object(raw_output)
    if payload:
        for start_key, end_key, note in (
            ("start_time", "end_time", None),
            ("start_sec", "end_sec", "legacy_keys_parse"),
        ):
            if start_key not in payload or end_key not in payload:
                continue
            try:
                start = float(payload[start_key])
                end = float(payload[end_key])
            except (TypeError, ValueError):
                continue
            if math.isfinite(start) and math.isfinite(end) and 0 <= start <= end:
                return [start, end], note
    if not isinstance(raw_output, str):
        return None, "TG_parse_failed"
    for pattern, converter, note in (
        (_TG_BRACKET, float, "bracket_parse"),
        (_TG_CLOCK_BRACKET, _clock_text_to_seconds, "clock_parse"),
        (_TG_LABELED_CLOCK, _clock_text_to_seconds, "labeled_clock_parse"),
        (_TG_LABELED, float, "labeled_parse"),
        (_TG_PROSE_CLOCK, _clock_text_to_seconds, "prose_clock_parse"),
        (_TG_PROSE, float, "prose_parse"),
    ):
        match = pattern.search(raw_output)
        if not match:
            continue
        start = converter(match.group(1))
        end = converter(match.group(2))
        if 0 <= start <= end:
            return [start, end], note
    numbers = _extract_numbers(raw_output)
    if len(numbers) >= 2 and math.isfinite(numbers[0]) and math.isfinite(numbers[1]) and 0 <= numbers[0] <= numbers[1]:
        return [numbers[0], numbers[1]], "fallback_numeric_parse"
    return None, "TG_parse_failed"


def parse_sg_prediction_for_scoring(
    raw_output: Any,
) -> tuple[dict[str, Any] | None, str | None]:
    """Parse canonical SG JSON plus the historical two-box JSON form."""

    def valid_box(box: Sequence[float]) -> bool:
        if len(box) != 4 or not all(math.isfinite(value) for value in box):
            return False
        x1, y1, x2, y2 = box
        return 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000

    payload = _extract_json_object(raw_output)
    if not payload:
        return None, "SG_json_parse_failed"
    if all(key in payload for key in ("start_time", "start_bbox", "end_time", "end_bbox")):
        try:
            start_time = float(payload["start_time"])
            end_time = float(payload["end_time"])
            start_bbox = [float(value) for value in payload["start_bbox"]]
            end_bbox = [float(value) for value in payload["end_bbox"]]
        except (TypeError, ValueError):
            pass
        else:
            if math.isfinite(start_time) and math.isfinite(end_time) and 0 <= start_time <= end_time and valid_box(start_bbox) and valid_box(end_bbox):
                return {
                    "start_time": start_time,
                    "start_bbox": start_bbox,
                    "end_time": end_time,
                    "end_bbox": end_bbox,
                }, None
    if isinstance(payload.get("boxes"), list):
        try:
            rows = payload["boxes"]
            if len(rows) != 2:
                raise ValueError("expected two boxes")
            start_time = float(rows[0]["timestamp_sec"])
            end_time = float(rows[1]["timestamp_sec"])
            start_bbox = [float(value) for value in rows[0]["bbox_xyxy_norm1000"]]
            end_bbox = [float(value) for value in rows[1]["bbox_xyxy_norm1000"]]
        except (KeyError, TypeError, ValueError):
            pass
        else:
            if math.isfinite(start_time) and math.isfinite(end_time) and 0 <= start_time <= end_time and valid_box(start_bbox) and valid_box(end_bbox):
                return {
                    "start_time": start_time,
                    "start_bbox": start_bbox,
                    "end_time": end_time,
                    "end_bbox": end_bbox,
                }, "legacy_boxes_parse"
    return None, "SG_schema_parse_failed"


def parse_qa_prediction_for_scoring(
    raw_output: Any,
) -> tuple[str | None, str | None]:
    if not isinstance(raw_output, str):
        return None, "QA_parse_failed"
    stripped = raw_output.strip().upper()
    exact = re.fullmatch(
        r"[\s\*\[\]\(\):`'\".-]*([A-E])[\s\*\[\]\(\):`'\".!,-]*",
        stripped,
    )
    if exact:
        answer = exact.group(1)
        return answer, None if stripped == answer else "decorated_letter_parse"
    matches = re.findall(r"(?<![A-Z])([A-E])(?![A-Z])", stripped)
    if matches:
        return matches[-1], "fallback_letter_parse"
    return None, "QA_parse_failed"


def parse_prediction_for_scoring(
    task: str,
    raw_output: Any,
) -> tuple[Any | None, str | None]:
    """Shared tolerant parser used by every formal evaluation runner."""

    if task == "TG":
        return parse_tg_prediction_for_scoring(raw_output)
    if task == "SG":
        return parse_sg_prediction_for_scoring(raw_output)
    if task == "QA":
        return parse_qa_prediction_for_scoring(raw_output)
    return None, f"unsupported_task_{task}"


def canonical_format_ok(task: str, raw_output: Any) -> bool:
    """Check exact output-contract compliance independently from scoring."""

    if not isinstance(raw_output, str) or not raw_output.strip():
        return False
    if task == "QA":
        return raw_output.strip().upper() in ANSWER_LETTERS
    try:
        payload = json.loads(raw_output.strip())
    except json.JSONDecodeError:
        return False
    parsed, error = parse_prediction(task, payload)
    return parsed is not None and error is None
