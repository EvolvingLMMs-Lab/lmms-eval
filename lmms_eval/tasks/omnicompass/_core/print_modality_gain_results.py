"""Exact function-level export of the internal evaluation implementation."""

from __future__ import annotations

import statistics
from typing import Any, Sequence

from .omnicompass_scoring import bbox_iou, temporal_iou

TASK_ORDER = ("TG", "SG", "QA")

CONDITION_ORDER = ("text_target", "reference_image")


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
