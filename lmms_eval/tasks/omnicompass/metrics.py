"""Official OmniCompass parsers, task metrics and paired consistency scores."""

from __future__ import annotations

import json
import math
import re
import statistics
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


GROUNDING_PASS_THRESHOLD = 0.5


def temporal_iou(prediction: list[float], ground_truth: list[float]) -> float:
    intersection = max(
        0.0,
        min(prediction[1], ground_truth[1]) - max(prediction[0], ground_truth[0]),
    )
    union = max(prediction[1], ground_truth[1]) - min(prediction[0], ground_truth[0])
    return intersection / union if union > 0 else 0.0


def bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def score_prediction(
    request: Mapping[str, Any],
    ground_truth: Mapping[str, Any],
    raw_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Parse and score one TG, SG, or QA response under the release contract."""

    task = str(request["task"])
    raw_output = raw_row.get("raw_output")
    if raw_output is None or not str(raw_output).strip():
        parsed = None
        parse_note = None
        parse_error = raw_row.get("error") or "empty_output"
    else:
        parsed, parse_note = parse_prediction_for_scoring(task, raw_output)
        parse_error = parse_note if parsed is None else None
        if parsed is None:
            parse_note = None

    score = 0.0
    correct = False
    protocol_diagnostics: list[str] = []
    metric = {
        "TG": "tIoU",
        "SG": "mean_keyframe_IoU",
        "QA": "accuracy",
    }.get(task)

    if parsed is not None and task == "TG":
        score = temporal_iou(
            [float(parsed[0]), float(parsed[1])],
            [
                float(
                    ground_truth.get(
                        "start_time",
                        ground_truth.get("start_sec"),
                    )
                ),
                float(
                    ground_truth.get(
                        "end_time",
                        ground_truth.get("end_sec"),
                    )
                ),
            ],
        )
        parse_error = None
    elif parsed is not None and task == "SG":
        expected_start = float(ground_truth.get("start_time", ground_truth.get("start_sec")))
        expected_end = float(ground_truth.get("end_time", ground_truth.get("end_sec")))
        if abs(float(parsed["start_time"]) - expected_start) > 1e-3 or abs(float(parsed["end_time"]) - expected_end) > 1e-3:
            protocol_diagnostics.append("SG_timestamps_do_not_match_given_interval")
        score = (bbox_iou(parsed["start_bbox"], ground_truth["start_bbox"]) + bbox_iou(parsed["end_bbox"], ground_truth["end_bbox"])) / 2.0
        parse_error = None
    elif parsed is not None and task == "QA":
        correct = str(parsed).upper() == str(ground_truth["answer"]).upper()
        score = 1.0 if correct else 0.0
        parse_error = None

    return {
        "request_id": request.get("request_id") or raw_row["request_id"],
        "video_id": request.get("video_id") or raw_row["video_id"],
        "track": request.get("track", "VL"),
        "task": task,
        "target_specification": (request.get("target_specification") or raw_row["target_specification"]),
        "raw_output": raw_output,
        "parsed": parsed,
        "parse_error": parse_error,
        "parse_note": parse_note,
        "protocol_diagnostics": protocol_diagnostics,
        "canonical_format": canonical_format_ok(task, raw_output),
        "score": score,
        "correct": correct if task == "QA" else None,
        "pass": (score > GROUNDING_PASS_THRESHOLD if task in {"TG", "SG"} else correct),
        "metric": metric,
        "ground_truth": dict(ground_truth),
        "http_status": raw_row.get("http_status"),
        "transport_error": raw_row.get("error_kind"),
    }


CONDITION_ORDER = ("text_target", "reference_image")


CONTENT_POLICY_ERROR_MARKERS = (
    "content_policy_violation",
    "content management policy",
    "content safety system",
    "content_filter",
    "contentfilter",
    "sensitive words",
    "prohibited use policy",
    "prohibited_content",
    "prompt_blocked",
    "data_inspection_failed",
    "datainspectionfailed",
    "inappropriate content",
)


REFUSAL_VALIDATION_ERRORS = {
    "content_policy_refusal",
    "finish_reason_content_filter",
}


def is_content_policy_error(value: Any) -> bool:
    text = str(value or "").lower()
    return any(marker in text for marker in CONTENT_POLICY_ERROR_MARKERS)


def has_policy_attempt(row: dict[str, Any]) -> bool:
    """True when any recorded attempt hit a provider content-policy error."""
    return any(is_content_policy_error(attempt.get("error")) for attempt in row.get("attempt_records") or [])


def is_final_policy_refusal(row: dict[str, Any]) -> bool:
    """True when the request's FINAL state is a provider content-policy refusal.

    Covers the API-exception path (error_kind / error text) and HTTP-200
    refusal text recorded as validation_error or a content_filter finish
    reason. Attempt history is deliberately NOT consulted here: a request that
    was policy-blocked on an early attempt but eventually succeeded has a
    usable answer and stays in the scoring denominator.
    """
    if row.get("error_kind") == "content_policy" or is_content_policy_error(row.get("error")):
        return True
    if str(row.get("validation_error") or "") in REFUSAL_VALIDATION_ERRORS:
        return True
    finish_reason = str((row.get("response_meta") or {}).get("finish_reason") or "")
    return finish_reason.lower() == "content_filter"


def is_terminal_policy_refusal(row: dict[str, Any]) -> bool:
    """True when the request never produced a usable answer due to policy.

    This is the final-state refusal OR a failed request (e.g. final timeout)
    whose attempt history already tripped the provider policy — such requests
    never delivered an answer and the triggering content is policy-flagged, so
    they are counted as safety-related exclusions rather than transport holes.
    """
    if is_final_policy_refusal(row):
        return True
    return row.get("error") is not None and has_policy_attempt(row)


def select_scoring_request_ids(
    raw_rows: Sequence[dict[str, Any]],
) -> tuple[set[str], set[str], set[str]]:
    """Split request IDs into (scoring, policy_excluded, transport_excluded).

    Official scoring uses successful API responses only: a row enters the
    denominator when its final state has no error and is not a final-state
    content-policy refusal. Successful responses that fail task parsing stay in
    the denominator with score 0. A request that eventually succeeded after a
    policy-flagged attempt also stays in the denominator.
    """
    scoring: set[str] = set()
    policy: set[str] = set()
    transport: set[str] = set()
    for row in raw_rows:
        request_id = str(row.get("request_id"))
        if row.get("error") is None and not is_final_policy_refusal(row):
            scoring.add(request_id)
        elif is_terminal_policy_refusal(row):
            policy.add(request_id)
        else:
            transport.add(request_id)
    return scoring, policy, transport


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def threshold_accuracy(values: Sequence[float], threshold: float) -> float | None:
    if not values:
        return None
    return sum(value >= threshold for value in values) / len(values)


def group_scores(
    parsed_rows: Sequence[dict[str, Any]],
    task: str,
    condition: str,
    scoring_ids: set[str] | None = None,
) -> list[float]:
    scores: list[float] = []
    for row in parsed_rows:
        if row.get("task") != task or row.get("target_specification") != condition:
            continue
        if scoring_ids is not None and str(row.get("request_id")) not in scoring_ids:
            continue
        score = row.get("score")
        if score is None:
            raise ValueError(f"{row.get('request_id')}: missing score")
        scores.append(float(score))
    return scores


def build_main_metrics(
    parsed_rows: Sequence[dict[str, Any]],
    scoring_ids: set[str] | None = None,
) -> dict[str, dict[str, float | None]]:
    output: dict[str, dict[str, float | None]] = {}
    selected_conditions = {str(row.get("target_specification")) for row in parsed_rows}
    unknown_conditions = selected_conditions - set(CONDITION_ORDER)
    if unknown_conditions:
        raise ValueError(f"unknown target specifications: {sorted(unknown_conditions)}")
    for condition in CONDITION_ORDER:
        if condition not in selected_conditions:
            continue
        tg_scores = group_scores(parsed_rows, "TG", condition, scoring_ids)
        sg_scores = group_scores(parsed_rows, "SG", condition, scoring_ids)
        qa_scores = group_scores(parsed_rows, "QA", condition, scoring_ids)
        tg_mean = mean(tg_scores)
        sg_mean = mean(sg_scores)
        qa_mean = mean(qa_scores)
        overall = mean([tg_mean, sg_mean, qa_mean]) if tg_mean is not None and sg_mean is not None and qa_mean is not None else None
        output[condition] = {
            "TG_A@0.3": threshold_accuracy(tg_scores, 0.3),
            "TG_A@0.5": threshold_accuracy(tg_scores, 0.5),
            "TG_A@0.7": threshold_accuracy(tg_scores, 0.7),
            "TG_m_tIoU": tg_mean,
            "SG_A@0.3": threshold_accuracy(sg_scores, 0.3),
            "SG_A@0.5": threshold_accuracy(sg_scores, 0.5),
            "SG_A@0.7": threshold_accuracy(sg_scores, 0.7),
            "SG_m_sIoU": sg_mean,
            "QA_accuracy": qa_mean,
            "Overall": overall,
        }
    return output


TASK_ORDER = ("TG", "SG", "QA")


def build_paired_cs(
    parsed_rows: Sequence[dict[str, Any]],
    scoring_ids: set[str],
) -> dict[str, Any] | None:
    by_key = {
        (
            str(row["video_id"]),
            str(row["task"]),
            str(row["target_specification"]),
        ): row
        for row in parsed_rows
    }
    conditions = {str(row["target_specification"]) for row in parsed_rows}
    if not set(CONDITION_ORDER).issubset(conditions):
        return None

    tasks: dict[str, Any] = {}
    for task in TASK_ORDER:
        video_ids = sorted({str(row["video_id"]) for row in parsed_rows if row.get("task") == task})
        values: list[float] = []
        parse_failure_pairs = 0
        excluded_pairs = 0
        for video_id in video_ids:
            text_row = by_key.get((video_id, task, "text_target"))
            image_row = by_key.get((video_id, task, "reference_image"))
            if text_row is None or image_row is None:
                excluded_pairs += 1
                continue
            if str(text_row["request_id"]) not in scoring_ids or str(image_row["request_id"]) not in scoring_ids:
                excluded_pairs += 1
                continue
            text_prediction = text_row.get("parsed")
            image_prediction = image_row.get("parsed")
            ground_truth = text_row["ground_truth"]
            if image_row["ground_truth"] != ground_truth:
                raise ValueError(f"{video_id}/{task}: paired ground truth mismatch")
            if text_prediction is None or image_prediction is None:
                parse_failure_pairs += 1
                values.append(0.0)
                continue
            if task == "QA":
                value = float(str(text_prediction).upper() == str(ground_truth).upper() and str(image_prediction).upper() == str(ground_truth).upper())
            elif task == "TG":
                target = [
                    float(ground_truth["start_time"]),
                    float(ground_truth["end_time"]),
                ]
                value = min(
                    temporal_iou(text_prediction, target),
                    temporal_iou(image_prediction, target),
                    temporal_iou(text_prediction, image_prediction),
                )
            else:
                value = statistics.mean(
                    min(
                        bbox_iou(text_prediction[key], ground_truth[key]),
                        bbox_iou(image_prediction[key], ground_truth[key]),
                        bbox_iou(text_prediction[key], image_prediction[key]),
                    )
                    for key in ("start_bbox", "end_bbox")
                )
            values.append(float(value))
        tasks[task] = {
            "value": statistics.mean(values) if values else None,
            "scored_pairs": len(values),
            "excluded_pairs": excluded_pairs,
            "parse_failure_pairs_scored_zero": parse_failure_pairs,
        }
    available = [tasks[task]["value"] for task in TASK_ORDER if tasks[task]["value"] is not None]
    return {
        "definition": ("paired successful-response intersection; QA requires both answers correct; TG/SG take the minimum of both GT overlaps and their mutual overlap; successful unparseable pairs score zero"),
        "tasks": tasks,
        "overall": statistics.mean(available) if available else None,
    }


SETTINGS = {"text": "text_target", "text_image": "reference_image"}


def scoring_ground_truth(question: Mapping[str, Any]) -> dict[str, Any]:
    """Restore SG's given interval, which is input rather than public gold."""
    gt = dict(question["ground_truth"])
    if question["task"] == "SG":
        start, end = question["given_interval"]
        gt.update(start_time=start, end_time=end)
    elif question["task"] not in {"TG", "QA"}:
        raise ValueError(f"Unknown task: {question['task']}")
    return gt


def make_request(question: Mapping[str, Any], setting: str) -> dict[str, Any]:
    if setting not in SETTINGS:
        raise ValueError(f"Unknown setting: {setting}")
    if question["track"] not in {"VL", "AV"}:
        raise ValueError(f"Unknown track: {question['track']}")
    return {
        "request_id": f"{question['question_id']}::{setting}",
        "video_id": question["video_id"],
        "track": question["track"],
        "task": question["task"],
        "target_specification": SETTINGS[setting],
        "ground_truth": scoring_ground_truth(question),
    }


def adapt_prediction(question: Mapping[str, Any], prediction: Mapping[str, Any]) -> dict[str, Any]:
    if prediction["question_id"] != question["question_id"]:
        raise ValueError("Prediction and question IDs do not match")
    request = make_request(question, prediction["setting"])
    # Retain final-state refusal, retry-history, and transport classification.
    return {
        **prediction,
        **{k: v for k, v in request.items() if k != "ground_truth"},
    }


def score_public_prediction(question: Mapping[str, Any], prediction: Mapping[str, Any]) -> dict[str, Any]:
    raw = adapt_prediction(question, prediction)
    request = make_request(question, prediction["setting"])
    scored = score_prediction(request, request["ground_truth"], raw)
    if question["task"] == "QA":
        # Existing paired-CS reporting expects a letter, not {"answer": ...}.
        scored["ground_truth"] = str(request["ground_truth"]["answer"]).upper()
    return scored


def evaluate_public_predictions(
    questions: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    settings: Sequence[str] = ("text", "text_image"),
) -> dict[str, Any]:
    """Evaluate the selected settings, reporting each track alone."""
    if not settings or len(set(settings)) != len(settings) or not set(settings) <= SETTINGS.keys():
        raise ValueError("Invalid setting selection")
    lookup = {q["question_id"]: q for q in questions}
    if len(lookup) != len(questions):
        raise ValueError("Duplicate question IDs")
    # The current shared CS reporter pairs one task per video. Reject multiple
    # instances explicitly until that reporter supports question-level pairing.
    keys = {(q["track"], q["video_id"], q["task"]) for q in questions}
    if len(keys) != len(questions):
        raise ValueError("Multiple questions for one track/video/task are unsupported")
    expected = {(qid, setting) for qid in lookup for setting in settings}
    seen: set[tuple[str, str]] = set()
    raw, parsed = [], []
    for row in predictions:
        key = (row["question_id"], row["setting"])
        if key not in expected or key in seen:
            raise ValueError(f"Unknown or duplicate prediction: {key}")
        seen.add(key)
        question = lookup[row["question_id"]]
        raw.append(adapt_prediction(question, row))
        parsed.append(score_public_prediction(question, row))
    if seen != expected:
        raise ValueError(f"Missing {len(expected - seen)} final prediction records")
    tracks = {}
    for track in sorted({q["track"] for q in questions}):
        track_raw = [r for r in raw if r["track"] == track]
        track_parsed = [r for r in parsed if r["track"] == track]
        scoring, policy, transport = select_scoring_request_ids(track_raw)
        tracks[track] = {
            "complete": not transport,
            "scoring_ids": sorted(scoring),
            "policy_excluded_ids": sorted(policy),
            "transport_excluded_ids": sorted(transport),
            "main_metrics": build_main_metrics(track_parsed, scoring),
            "paired_cs": build_paired_cs(track_parsed, scoring) if len(settings) == 2 else None,
        }
    return {"parsed_predictions": parsed, "tracks": tracks}
