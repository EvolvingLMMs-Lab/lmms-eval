"""Shared scoring for every maintained OmniCompass evaluation runner."""

from __future__ import annotations

from typing import Any, Mapping

from .omnicompass_release import canonical_format_ok, parse_prediction_for_scoring

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
