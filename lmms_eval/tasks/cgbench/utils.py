import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.media_resolver import resolve_media_reference

Document = Mapping[str, Any]
TaskKwargs = Mapping[str, Any]
MetricRecord = dict[str, Any]


def _video_reference(doc: Document) -> str:
    return f"{doc['video_uid']}.mp4"


def _resolve_video(doc: Document) -> str:
    return resolve_media_reference(
        _video_reference(doc),
        media_type="video",
        cache_dir="cg_videos_720p",
        env_vars=("CGBENCH_VIDEO_DIR", "CGBENCH_ROOT"),
        extra_subdirs=("cg_videos_720p", "videos"),
    )


def cgbench_doc_to_visual(doc: Document) -> list[str]:
    """Resolve the local video for one CG-Bench example."""
    video_path = _resolve_video(doc)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"CG-Bench video not found: {video_path}. Accept the dataset terms on Hugging Face and set CGBENCH_VIDEO_DIR to the extracted video directory if needed.")
    return [video_path]


def _format_question(doc: Document) -> str:
    choices = "\n".join(f"{chr(65 + index)}. {choice}" for index, choice in enumerate(doc["choices"]))
    return f"{doc['question']}\n{choices}"


def cgbench_doc_to_text(doc: Document, lmms_eval_specific_kwargs: TaskKwargs | None = None) -> str:
    """Format the video-only multiple-choice prompt."""
    kwargs = lmms_eval_specific_kwargs or {}
    instruction = "Select the best answer to the following multiple-choice question based on the video."
    return f"{kwargs.get('pre_prompt', '')}{instruction}\n{_format_question(doc)}{kwargs.get('post_prompt', '')}"


def _parse_srt_timestamp(value: str) -> float:
    hours, minutes, seconds = value.strip().replace(".", ",").split(":")
    seconds, milliseconds = seconds.split(",")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000


def _read_srt(path: str | Path) -> list[tuple[float, float, str]]:
    entries = []
    content = Path(path).read_text(encoding="utf-8", errors="replace")
    for block in re.split(r"\r?\n\r?\n", content.strip()):
        lines = block.splitlines()
        time_line = next((line for line in lines if " --> " in line), None)
        if time_line is None:
            continue
        start, end = time_line.split(" --> ", 1)
        time_index = lines.index(time_line)
        entries.append((_parse_srt_timestamp(start), _parse_srt_timestamp(end), " ".join(lines[time_index + 1 :])))
    return entries


def _resolve_subtitle(doc: Document) -> str:
    return resolve_media_reference(
        f"{doc['video_uid']}.srt",
        media_type="video",
        cache_dir="cg_videos_720p",
        env_vars=("CGBENCH_SUBTITLE_DIR", "CGBENCH_ROOT"),
        extra_subdirs=("cg_subtitles", "subtitles"),
    )


def _sampled_subtitles(doc: Document, frame_num: int) -> str:
    subtitle_path = _resolve_subtitle(doc)
    if not os.path.exists(subtitle_path):
        return "No subtitles available."

    entries = _read_srt(subtitle_path)
    if not entries:
        return "No subtitles available."

    video_path = _resolve_video(doc)
    capture = cv2.VideoCapture(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if fps <= 0 or frame_count <= 0 or frame_num == -1:
        return "\n".join(text for _, _, text in entries)

    sample_count = max(1, min(int(frame_num), frame_count))
    timestamps = [index * (frame_count - 1) / max(sample_count - 1, 1) / fps for index in range(sample_count)]
    selected = []
    for start, end, text in entries:
        if any(start <= timestamp < end for timestamp in timestamps):
            selected.append(text)
    return "\n".join(selected) if selected else "No subtitles available."


def cgbench_doc_to_text_subtitle(doc: Document, lmms_eval_specific_kwargs: TaskKwargs | None = None) -> str:
    """Format the multiple-choice prompt with subtitles aligned to sampled frames."""
    kwargs = lmms_eval_specific_kwargs or {}
    subtitles = _sampled_subtitles(doc, kwargs.get("frame_num", 32))
    instruction = "Select the best answer to the following multiple-choice question based on the video and subtitles."
    return f"{kwargs.get('pre_prompt', '')}This video's subtitles are listed below:\n{subtitles}\n{instruction}\n{_format_question(doc)}{kwargs.get('post_prompt', '')}"


def extract_characters_regex(response: str) -> str:
    """Extract one valid CG-Bench option letter from a model response."""
    from lmms_eval.tasks._task_utils.mcq_extract import extract_mcq_answer

    return extract_mcq_answer(response or "", choices=[chr(65 + index) for index in range(14)])


def cgbench_process_results(doc: Document, results: Sequence[str]) -> dict[str, MetricRecord]:
    """Parse a prediction and emit the CG-Bench accuracy record."""
    prediction = results[0] if results else ""
    parsed_prediction = extract_characters_regex(prediction)
    answer = str(doc["right_answer"]).strip().upper()
    return {
        "cgbench_accuracy": {
            "question_id": doc["qid"],
            "video_uid": doc["video_uid"],
            "duration": doc["duration"],
            "category": doc["domain"],
            "sub_category": doc["sub_category"],
            "pred_answer": parsed_prediction,
            "answer": answer,
            "score": float(parsed_prediction == answer),
        }
    }


def cgbench_aggregate_results(results: Sequence[MetricRecord]) -> float:
    """Aggregate CG-Bench exact-match accuracy as a percentage."""
    if not results:
        return 0.0
    category_scores = defaultdict(list)
    subcategory_scores = defaultdict(list)
    for result in results:
        category_scores[result["category"]].append(result["score"])
        subcategory_scores[result["sub_category"]].append(result["score"])
    for name, scores in sorted(category_scores.items()):
        eval_logger.info(f"CG-Bench/{name}: {100 * sum(scores) / len(scores):.2f}")
    for name, scores in sorted(subcategory_scores.items()):
        eval_logger.debug(f"CG-Bench/{name}: {100 * sum(scores) / len(scores):.2f}")
    return 100.0 * sum(result["score"] for result in results) / len(results)
