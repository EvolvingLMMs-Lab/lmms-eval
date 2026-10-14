"""Model-visible Prompt contract for OmniCompass AV frame-and-audio input."""

from __future__ import annotations

from typing import Any, Mapping

from .omnicompass_frame_prompt import (
    duration_text,
    qa_body,
    spatial_body,
    temporal_body,
)

AV_FRAME_PROMPT_REVISION = "paper-av-frame-audio-visible-timestamp-hybrid-xyxy-ltrb-v1"
AV_FRAME_PROMPT_REVISION_NO_TIMESTAMP = "paper-av-frame-audio-no-visible-timestamp-hybrid-xyxy-ltrb-v1"
AV_FRAME_INPUT_PROTOCOL = "uniform_sampled_frames_audio_duration_in_prefix_reference_last_visible_timestamp_text_hybrid_xyxy_ltrb_v1"
AV_FRAME_INPUT_PROTOCOL_NO_TIMESTAMP = "uniform_sampled_frames_audio_duration_in_prefix_reference_last_no_visible_timestamp_text_hybrid_xyxy_ltrb_v1"


def input_prefix(
    target_specification: str,
    duration: str,
    *,
    visible_frame_timestamps: bool = True,
) -> str:
    if target_specification == "reference_image":
        prefix = f"You are given a sequence of frames uniformly sampled across a {duration}-second video and the corresponding audio, followed by a reference image. "
        if visible_frame_timestamps:
            prefix += "Each frame is preceded by its timestamp in seconds. "
        return prefix + "Complete the task using the video frames, audio, and reference image."
    if target_specification == "text_target":
        prefix = f"You are given a sequence of frames uniformly sampled across a {duration}-second video and the corresponding audio. "
        if visible_frame_timestamps:
            prefix += "Each frame is preceded by its timestamp in seconds. "
        return prefix + "Complete the task using the video frames and audio."
    raise ValueError(f"unsupported target specification: {target_specification!r}")


def build_av_frame_prompt(
    *,
    task: str,
    target_specification: str,
    question: str,
    ground_truth: Mapping[str, Any],
    video_manifest: Mapping[str, Any],
    visible_frame_timestamps: bool = True,
) -> str:
    if task == "TG":
        body = temporal_body(question)
    elif task == "SG":
        body = spatial_body(question, ground_truth)
    elif task == "QA":
        body = qa_body(question, ground_truth)
    else:
        raise ValueError(f"unsupported task: {task!r}")
    return (
        input_prefix(
            target_specification,
            duration_text(video_manifest),
            visible_frame_timestamps=visible_frame_timestamps,
        )
        + "\n\n"
        + body
    )
