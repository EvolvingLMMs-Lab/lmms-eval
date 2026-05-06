"""
HD-EPIC benchmark integration for lmms-eval.

HD-EPIC is a video QA benchmark covering egocentric kitchen activities.
Each question is multiple-choice, referencing one or more video clips with
optional start/end timestamps.

Dataset record format (after conversion via hd_epic_to_hf.py):
  {
    "question_id":  str,          # unique ID, e.g. "action_localization_0042"
    "task_type":    str,          # e.g. "action_localization"
    "question":     str,          # question text (may contain <TIME> / <BBOX> tags)
    "choices":      List[str],    # answer options
    "correct_idx":  int,          # 0-based index into `choices`
    "video_ids":    List[str],    # participant-prefixed IDs, e.g. ["P01-20240427-151808"]
    "input_labels": List[str],    # e.g. ["video 1"] or ["video 1", "video 2"]
    "start_times":  List[float],  # seconds; -1 = start of full clip
    "end_times":    List[float],  # seconds; -1 = end of full clip
    "video_dir":    str,          # base directory that holds <pid>/<vid>.mp4 files
  }

Environment variable HD_EPIC_VIDEO_DIR overrides the `video_dir` field at runtime.
"""

from __future__ import annotations

import datetime
import os
import os.path as osp
import re
import subprocess
import tempfile
from typing import List, Optional

from loguru import logger as eval_logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHOICE_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# HD-EPIC native recording resolution (Project Aria RGB camera). BBOX coords in
# the question JSON are in this pixel space and are normalised to 1000x1000 for
# the prompt -- matching the format every Gemini-style model expects.
# Override via env var HD_EPIC_ORIG_RES if your data has a different resolution.
_DEFAULT_ORIG_RES = 1408

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _secs_from_time_str(s: str) -> float:
    """
    Parse 'HH:MM:SS.fff' (or 'HH:MM:SS', or short variants) -> float seconds.
    Returns -1.0 on failure rather than raising, since we never want a single
    malformed timestamp in one question to crash an entire eval run.
    """
    try:
        h_str, m_str, s_str = s.split(":", 2)
        return int(h_str) * 3600 + int(m_str) * 60 + float(s_str)
    except (ValueError, AttributeError, TypeError):
        return -1.0


def _resolve_video_dir(doc: dict) -> str:
    """Return video root directory: env-var > doc field > cwd."""
    return os.environ.get(
        "HD_EPIC_VIDEO_DIR",
        doc.get("video_dir", os.getcwd()),
    )


def _video_path(video_id: str, video_dir: str) -> str:
    """Return the .mp4 path for a given video_id."""
    pid = video_id.split("-")[0]
    return osp.join(video_dir, pid, f"{video_id}.mp4")


def _extract_clip(video_id: str, start_s: float, end_s: float, video_dir: str) -> str:
    """Extract a time-trimmed subclip from a source video and return its path.

    Uses a two-pass ffmpeg seek strategy for frame-accurate boundaries at
    practical speed: a fast keyframe-aligned input seek jumps to ~2 seconds
    before the target, then a short frame-accurate output seek covers the
    remaining offset. This avoids the 10-20× slowdown of pure output seek
    on long videos while producing the same frame-accurate result.

    Extracted clips are cached in the system temp directory under the name
    ``hd_epic_<video_id>_<start>_<end>.mp4``. If the cache file already
    exists it is returned immediately without re-extracting.

    Args:
        video_id: Participant-prefixed video ID, e.g. ``"P01-20240427-151808"``.
        start_s:  Clip start time in seconds. Pass ``-1`` to use the full video.
        end_s:    Clip end time in seconds.   Pass ``-1`` to use the full video.
        video_dir: Root directory containing ``<pid>/<video_id>.mp4`` files.
                   Overridden at runtime by the ``HD_EPIC_VIDEO_DIR`` env var
                   (handled upstream by ``_resolve_video_dir``).

    Returns:
        Absolute path to the extracted ``.mp4`` clip, or the path to the
        original source file when no trimming is needed (``start_s == end_s
        == -1``) or when ffmpeg fails (falls back gracefully with a logged
        error rather than raising).
    """
    src = _video_path(video_id, video_dir)

    if start_s == -1 and end_s == -1:
        return src

    if not osp.exists(src):
        eval_logger.warning(f"HD-EPIC: video not found: {src}")
        return src

    tmp_dir = tempfile.gettempdir()
    out_fn = osp.join(tmp_dir, f"hd_epic_{video_id}_{int(start_s)}_{int(end_s)}.mp4")
    if osp.exists(out_fn):
        return out_fn

    if end_s <= start_s:
        end_s = start_s + 1.0
    dur = end_s - start_s

    # Two-pass seek: coarse input seek to ~2s before target (fast, jumps via
    # keyframe), then fine output seek for the remaining offset (frame-
    # accurate). Equivalent timing precision to pure output seek but orders
    # of magnitude faster on long videos because the keyframe jump skips
    # most of the decode.
    coarse_start = max(0.0, start_s - 2.0)
    fine_start = start_s - coarse_start  # 0 to 2 seconds

    cmd = (
        f"ffmpeg -y -hide_banner -loglevel error "
        f"-ss {coarse_start} "                       # input seek (fast)
        f"-i {src} "
        f"-ss {fine_start} -t {dur} "                # output seek (precise)
        f"-c:v libx264 -preset ultrafast -crf 23 "
        f"{out_fn}"
    )
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as exc:
        eval_logger.error(f"HD-EPIC: ffmpeg clip extraction failed: {exc}")
        return src

    return out_fn


def _parse_tags(text: str, video_ids: List[str], start_times: List[float], input_labels: List[str] = None) -> str:
    """
    Replace <TIME HH:MM:SS.fff VID_ID> with a human-readable timestamp
    and <BBOX y1 x1 y2 x2> with a formatted string.
    Timestamps are made relative to the clip start.

    NOTE on lookup keys: HD-EPIC tags reference the *input label* (e.g.
    "video 1"), not the underlying video ID. So we build the offset map
    keyed by label, falling back to the video ID for backward compatibility.
    """
    if input_labels is None:
        input_labels = [f"video {i + 1}" for i in range(len(video_ids))]

    input_start = {}
    for label, vid, st in zip(input_labels, video_ids, start_times):
        input_start[label] = max(0.0, st)
        input_start[vid] = max(0.0, st)  # backward-compat fallback

    def _time_repl(m: re.Match) -> str:
        raw_secs = _secs_from_time_str(m.group(1))
        key = m.group(2).strip()
        offset = input_start.get(key, 0.0)
        rel = max(0.0, raw_secs - offset)
        if rel >= 3600:
            return datetime.time(
                int(rel // 3600),
                int((rel % 3600) // 60),
                int(rel % 60),
            ).strftime("%H:%M:%S")
        return datetime.time(0, int(rel // 60), int(rel % 60)).strftime("%M:%S")

    text = re.sub(r"<TIME\s+([\d:.]+)\s+(.+?)>", _time_repl, text)

    orig_res = float(os.environ.get("HD_EPIC_ORIG_RES", _DEFAULT_ORIG_RES))

    def _bbox_repl(m: re.Match) -> str:
        # Normalise pixel coords (1408x1408 native) into a 1000x1000 frame --
        # this matches what HD-EPIC's base_model.py does and what Gemini-style
        # models expect ("(ymin, xmin, ymax, xmax) on a 1000x1000 image").
        coords = [int(float(c) / orig_res * 1000) for c in m.groups()]
        return f"({', '.join(map(str, coords))})"

    text = re.sub(
        r"<BBOX\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*>",
        _bbox_repl,
        text,
    )
    return text


def _format_choice(choice, video_ids, start_times, input_labels=None):
    if not isinstance(choice, str):
        choice = str(choice)
    return _parse_tags(choice, video_ids, start_times, input_labels)


def _format_question_text(doc: dict) -> str:
    """Build the formatted question string (with answer choices)."""
    q_text = doc["question"]
    video_ids: List[str] = doc.get("video_ids", [])
    start_times: List[float] = doc.get("start_times", [])
    input_labels: List[str] = doc.get("input_labels", [])

    q_text = _parse_tags(q_text, video_ids, start_times, input_labels)

    choices_str = " ".join(f"({CHOICE_LETTERS[i]}) {_format_choice(c, video_ids, start_times, input_labels)}." for i, c in enumerate(doc["choices"]))
    return f"Question: {q_text}. Answers: {choices_str} Correct: "


# ---------------------------------------------------------------------------
# lmms-eval API functions
# ---------------------------------------------------------------------------


def hd_epic_doc_to_visual(doc: dict) -> List[str]:
    """
    Legacy simple-model path: return a list of local video file paths
    (one per input video, trimmed to the required time window).
    """
    video_dir = _resolve_video_dir(doc)
    video_ids: List[str] = doc.get("video_ids", [])
    start_times: List[float] = doc.get("start_times", [])
    end_times: List[float] = doc.get("end_times", [])

    paths = []
    for vid, st, et in zip(video_ids, start_times, end_times):
        p = _extract_clip(vid, st, et, video_dir)
        paths.append(p)
    return paths


def hd_epic_doc_to_text(doc: dict, lmms_eval_specific_kwargs: Optional[dict] = None) -> str:
    """
    Legacy simple-model path: return the formatted question string.
    The system instruction is prepended by the model, so we only return
    the question + answer choices here.
    """
    return _format_question_text(doc)


def hd_epic_doc_to_messages(doc: dict, lmms_eval_specific_kwargs: Optional[dict] = None) -> List[dict]:
    """
    Chat-model path (recommended). Produces a structured message list:

      [
        { "role": "system",  "content": [{"type": "text", "text": sys_prompt}] },
        { "role": "user",    "content": [
            {"type": "video", "url": <path>},
            ...
            {"type": "text",  "text": <question + choices>}
          ]
        }
      ]

    Multiple videos are prepended in the order they appear in the question.
    """
    sys_prompt = (
        "You are an expert video analyzer, and your job is to answer the multiple "
        "choice question by giving only the letter identifying the answer. Do not "
        "give any other information. For example, acceptable answers are 'A' or 'B' "
        "or 'C' etc.. You must give an answer, even if you are not sure. Bounding "
        "boxes are in the format (ymin, xmin, ymax, xmax) relative to an image size "
        "of 1000x1000."
    )

    video_dir = _resolve_video_dir(doc)
    video_ids: List[str] = doc.get("video_ids", [])
    input_labels: List[str] = doc.get("input_labels", [])
    start_times: List[float] = doc.get("start_times", [])
    end_times: List[float] = doc.get("end_times", [])

    user_content = []
    for i, (vid, st, et) in enumerate(zip(video_ids, start_times, end_times)):
        label = input_labels[i] if i < len(input_labels) else f"video {i + 1}"
        path = _extract_clip(vid, st, et, video_dir)
        user_content.append({"type": "text", "text": f"{label}: "})
        user_content.append({"type": "video", "url": path})

    user_content.append({"type": "text", "text": _format_question_text(doc)})

    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": sys_prompt}],
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def hd_epic_process_results(doc: dict, results: List[str]) -> dict:
    """
    Parse the model's free-text response and compare it to `correct_idx`.

    Returns a dict with the 'accuracy' metric key.
    """
    if not results:
        return {"accuracy": 0.0}

    response: str = results[0].strip()

    # Extract the first uppercase letter A-Z from the response
    match = re.search(r"[A-Z]", response.upper())
    if match is None:
        pred_idx = -1
    else:
        pred_idx = ord(match.group(0)) - ord("A")
        n_choices = len(doc["choices"])
        if pred_idx < 0 or pred_idx >= n_choices:
            pred_idx = -1

    correct_idx = int(doc.get("correct_idx", -1))
    is_correct = 1.0 if (pred_idx == correct_idx and correct_idx >= 0) else 0.0

    return {"accuracy": is_correct}


def hd_epic_doc_to_target(doc: dict) -> str:
    """
    Return the correct answer letter for a given question doc.
    lmms-eval requires this for every task even with output_type: generate_until.
    """
    correct_idx = int(doc.get("correct_idx", 0))
    return CHOICE_LETTERS[correct_idx]


def hd_epic_aggregate_accuracy(results: List[float]) -> float:
    """Aggregate accuracy across all results."""
    if not results:
        return 0.0
    return sum(results) / len(results)


# ---------------------------------------------------------------------------
# Dataset filtering helpers
# ---------------------------------------------------------------------------
#
# lmms-eval supports a `process_docs:` field in the task YAML that runs once
# on the full dataset before evaluation. We use it to filter records by
# `task_type`, so a single combined JSONL can serve every subtask.
#
# The factory below returns a closure for a given task_type. Each per-task
# YAML references its own filter function, e.g.:
#
#   process_docs: !function utils.filter_action_recognition
#
# All filter_<task_type> functions are auto-generated below at import time.


def _make_filter(task_type: str):
    """Return a function that keeps only rows whose task_type matches."""

    def _filter(dataset):
        return dataset.filter(lambda doc: doc.get("task_type") == task_type)

    _filter.__name__ = f"filter_{task_type}"
    return _filter


# Auto-register one filter function per HD-EPIC task type so that they can
# be referenced from YAML via `!function utils.filter_<task_type>`.
# Source of truth: hd-epic-annotations/vqa-benchmark/ (30 official prototypes).
_HD_EPIC_TASK_TYPES = [
    # Recipe (8)
    "recipe_recipe_recognition",
    "recipe_multi_recipe_recognition",
    "recipe_multi_step_localization",
    "recipe_step_localization",
    "recipe_prep_localization",
    "recipe_step_recognition",
    "recipe_rough_step_localization",
    "recipe_following_activity_recognition",
    # Ingredient (6)
    "ingredient_ingredient_retrieval",
    "ingredient_ingredient_weight",
    "ingredient_ingredients_order",
    "ingredient_ingredient_adding_localization",
    "ingredient_ingredient_recognition",
    "ingredient_exact_ingredient_recognition",
    # Nutrition (3)
    "nutrition_image_nutrition_estimation",
    "nutrition_nutrition_change",
    "nutrition_video_nutrition_estimation",
    # Fine-grained Actions (4)
    "fine_grained_action_recognition",
    "fine_grained_how_recognition",
    "fine_grained_why_recognition",
    "fine_grained_action_localization",
    # 3D Perception (4)
    "3d_perception_fixture_location",
    "3d_perception_object_location",
    "3d_perception_object_contents_retrieval",
    "3d_perception_fixture_interaction_counting",
    # Object Motion (3)
    "object_motion_object_movement_itinerary",
    "object_motion_object_movement_counting",
    "object_motion_stationary_object_localization",
    # Gaze (2)
    "gaze_gaze_estimation",
    "gaze_interaction_anticipation",
]

for _tt in _HD_EPIC_TASK_TYPES:
    globals()[f"filter_{_tt}"] = _make_filter(_tt)

