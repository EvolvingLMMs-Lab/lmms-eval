"""Adapt the public question schema to the existing prompt and scoring code.

Predictions identify one final response by (question_id, setting), where setting
is "text" or "text_image". Transport/policy metadata must be retained; an empty
completed response is different from a failed request. This module does not
perform inference or change the benchmark's scoring policies.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .omnicompass_av_frame_prompt import build_av_frame_prompt
from .omnicompass_frame_prompt import build_frame_prompt
from .omnicompass_scoring import score_prediction
from .print_frame_eval_results import build_main_metrics, select_scoring_request_ids
from .print_modality_gain_results import build_paired_cs

SETTINGS = {"text": "text_target", "text_image": "reference_image"}


def uses_audio(track: str, mode: str = "auto") -> bool:
    if track not in {"VL", "AV"} or mode not in {"auto", "on", "off"}:
        raise ValueError("Invalid track or audio mode")
    return track == "AV" if mode == "auto" else mode == "on"


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


def render_prompt(
    question: Mapping[str, Any],
    setting: str,
    duration_seconds: float,
    use_audio: bool | None = None,
) -> str:
    request = make_request(question, setting)
    gt = dict(request["ground_truth"])
    if question["task"] == "QA":
        gt["options"] = question["options"]
    if use_audio is None:
        use_audio = uses_audio(question["track"])
    builder = build_av_frame_prompt if use_audio else build_frame_prompt
    return builder(
        task=question["task"],
        target_specification=request["target_specification"],
        question=question["question"][setting],
        ground_truth=gt,
        video_manifest={"duration_sec": duration_seconds},
    )


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
