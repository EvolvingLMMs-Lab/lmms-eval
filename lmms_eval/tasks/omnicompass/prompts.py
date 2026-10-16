"""Official paired text and image prompts for VL and AV."""

from __future__ import annotations

from typing import Any, Mapping

from .metrics import ANSWER_LETTERS, format_number, make_request

FRAME_PROMPT_REVISION = "paper-frame-visible-timestamp-hybrid-xyxy-ltrb-v1"


FRAME_PROMPT_REVISION_NO_TIMESTAMP = "paper-frame-no-visible-timestamp-hybrid-xyxy-ltrb-v1"


FRAME_INPUT_PROTOCOL = "uniform_sampled_frames_duration_in_prefix_reference_last_visible_timestamp_text_hybrid_xyxy_ltrb_v1"


FRAME_INPUT_PROTOCOL_NO_TIMESTAMP = "uniform_sampled_frames_duration_in_prefix_reference_last_no_visible_timestamp_text_hybrid_xyxy_ltrb_v1"


def duration_text(video_manifest: Mapping[str, Any]) -> str:
    return format_number(round(float(video_manifest["duration_sec"]), 1))


def input_prefix(
    target_specification: str,
    duration: str,
    *,
    visible_frame_timestamps: bool = True,
) -> str:
    if target_specification == "reference_image":
        prefix = f"You are given a sequence of frames uniformly sampled across a {duration}-second video, followed by a reference image. "
        if visible_frame_timestamps:
            prefix += "Each video frame is preceded by its timestamp in seconds. "
        return prefix + "Complete the task based on the visual content of the video frames and the reference image."
    if target_specification == "text_target":
        prefix = f"You are given a sequence of frames uniformly sampled across a {duration}-second video. "
        if visible_frame_timestamps:
            prefix += "Each video frame is preceded by its timestamp in seconds. "
        return prefix + "Complete the task based on the visual content of these video frames."
    raise ValueError(f"unsupported target specification: {target_specification!r}")


def temporal_body(question: str) -> str:
    return f'Question: {question}\n\nOutput format: Answer with only this JSON object, where start_time and end_time are measured in seconds from the beginning of the video:\n\n{{"start_time": <number>,\n "end_time": <number>}}'


def spatial_body(question: str, ground_truth: Mapping[str, Any]) -> str:
    start = format_number(ground_truth["start_time"])
    end = format_number(ground_truth["end_time"])
    return (
        f"Question: {question}\n\n"
        f"The relevant interval starts at {start} seconds and ends at {end} "
        "seconds. Locate the target described in the question at each of "
        "these two timestamps and report one bounding box for each frame.\n"
        "Write each box in [x1, y1, x2, y2] order, corresponding to "
        "[left, top, right, bottom]. Normalize all coordinates from 0 to "
        "1000, with the top-left of the frame as the origin.\n\n"
        "Output format: Return only this JSON object:\n\n"
        f'{{"start_time": {start},\n'
        ' "start_bbox": [<x1>, <y1>, <x2>, <y2>],\n'
        f' "end_time": {end},\n'
        ' "end_bbox": [<x1>, <y1>, <x2>, <y2>]}'
    )


def qa_body(question: str, ground_truth: Mapping[str, Any]) -> str:
    options = ground_truth.get("options")
    if not isinstance(options, Mapping):
        raise ValueError("QA ground truth requires ordered A-E options")
    if list(options.keys()) != list(ANSWER_LETTERS):
        raise ValueError("QA options must use the ordered keys A-E")
    option_lines = "\n".join(f"{letter}. {options[letter]}" for letter in ANSWER_LETTERS)
    return f"Question: {question}\n\n{option_lines}\n\nYou must choose the most appropriate answer from the five options\n(A, B, C, D, or E). If uncertain, make your best choice based on the\nprovided inputs. Output only the option letter."


def build_frame_prompt(
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


AV_FRAME_PROMPT_REVISION = "paper-av-frame-audio-visible-timestamp-hybrid-xyxy-ltrb-v1"


AV_FRAME_PROMPT_REVISION_NO_TIMESTAMP = "paper-av-frame-audio-no-visible-timestamp-hybrid-xyxy-ltrb-v1"


AV_FRAME_INPUT_PROTOCOL = "uniform_sampled_frames_audio_duration_in_prefix_reference_last_visible_timestamp_text_hybrid_xyxy_ltrb_v1"


AV_FRAME_INPUT_PROTOCOL_NO_TIMESTAMP = "uniform_sampled_frames_audio_duration_in_prefix_reference_last_no_visible_timestamp_text_hybrid_xyxy_ltrb_v1"


def audio_input_prefix(
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
        audio_input_prefix(
            target_specification,
            duration_text(video_manifest),
            visible_frame_timestamps=visible_frame_timestamps,
        )
        + "\n\n"
        + body
    )


SETTINGS = {"text": "text_target", "text_image": "reference_image"}


def uses_audio(track: str, mode: str = "auto") -> bool:
    if track not in {"VL", "AV"} or mode not in {"auto", "on", "off"}:
        raise ValueError("Invalid track or audio mode")
    return track == "AV" if mode == "auto" else mode == "on"


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
