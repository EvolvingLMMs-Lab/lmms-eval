import glob
import os
import re
from pathlib import Path

from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.media_resolver import resolve_media_reference

_HF_REPO = "lmms-lab/EgoLife"
_PARTICIPANT = "A1_JAKE"
_VIDEO_EXTENSIONS = ("mp4", "MP4", "mkv", "webm", "mov")


def _parse_time_str(time_str: str) -> int | None:
    s = str(time_str).strip()
    if not s.isdigit():
        return None
    try:
        if len(s) == 8:
            h = int(s[0:2])
            m = int(s[2:4])
            sec = int(s[4:6])
            frac = int(s[6:8])
            return h * 3600 + m * 60 + sec + frac / 100
        if len(s) == 6:
            h = int(s[0:2])
            m = int(s[2:4])
            sec = int(s[4:6])
            return h * 3600 + m * 60 + sec
        return int(s)
    except ValueError:
        return None


def _time_to_seconds(time_obj) -> int | None:
    if isinstance(time_obj, dict):
        t = time_obj.get("time", "")
        return _parse_time_str(t)
    return _parse_time_str(time_obj)


def _candidate_video_dirs(participant: str = _PARTICIPANT) -> list[Path]:
    paths: list[Path] = []
    explicit = os.getenv("EGOLIFEQA_VIDEO_DIR", "").strip()
    if explicit:
        paths.append(Path(os.path.expanduser(explicit)))
    explicit_cache = os.getenv("EGOLIFEQA_CACHE_DIR", "").strip()
    if explicit_cache:
        paths.append(Path(os.path.expanduser(explicit_cache)))
    hf_home = Path(os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface")))
    paths.append(hf_home / "egolifeqa")
    paths.append(hf_home / "egolife")
    # hf hub snapshot location for lmms-lab/EgoLife
    hub_base = hf_home / "hub" / "datasets--lmms-lab--EgoLife"
    if hub_base.exists():
        for snap in (hub_base / "snapshots").glob("*"):
            paths.append(snap / participant)
            paths.append(snap)
    return paths


def _find_nearest_video(participant: str, day: str, query_time: str) -> str | None:
    target_sec = _parse_time_str(query_time)
    if target_sec is None:
        return None
    day = str(day).strip()
    search_dirs = _candidate_video_dirs(participant)
    # Also try resolving via media_resolver for direct filename guess
    # Guess filename: DAYx_PARTICIPANT_HHMMSS00.mp4 floored to 30s
    for root in search_dirs:
        # Pattern: DAY*/DAY*_PARTICIPANT*.mp4 sorted
        pattern = str(root / participant / day / f"{day}_{participant}_*.mp4")
        candidates = glob.glob(pattern)
        if not candidates:
            pattern2 = str(root / day / f"{day}_{participant}_*.mp4")
            candidates = glob.glob(pattern2)
        if not candidates:
            pattern3 = str(root / f"{day}_{participant}_*.mp4")
            candidates = glob.glob(pattern3)
        if candidates:
            best = None
            best_dist = float("inf")
            for cand in candidates:
                m = re.search(r"_(\d{8})(?:\.mp4)?$", cand)
                if not m:
                    m = re.search(r"_(\d{6})(?:\.mp4)?$", cand)
                if m:
                    sec = _parse_time_str(m.group(1))
                    if sec is not None:
                        dist = abs(sec - target_sec)
                        if dist < best_dist:
                            best_dist = dist
                            best = cand
            if best:
                return best
    # Fallback: try media_resolver direct reference without existence check
    guess = f"{day}_{participant}_{str(query_time)[:6]}0000"
    resolved = resolve_media_reference(guess, media_type="video", cache_dir="egolifeqa", env_vars=("EGOLIFEQA_VIDEO_DIR", "EGOLIFEQA_CACHE_DIR"))
    if isinstance(resolved, str) and os.path.exists(resolved):
        return resolved
    return None


def egolifeqa_doc_to_visual(doc):
    participant = str(doc.get("participant", _PARTICIPANT)).strip() or _PARTICIPANT
    query_time = doc.get("query_time", {})
    day = ""
    time_str = ""
    if isinstance(query_time, dict):
        day = query_time.get("date", "")
        time_str = query_time.get("time", "")
    else:
        day = str(doc.get("day", "DAY1"))
        time_str = str(query_time)
    if not day:
        day = "DAY1"
    if not time_str:
        time_str = "11000000"

    # Try direct video field if present
    for key in ("video", "video_path", "video_file", "clip_path"):
        val = doc.get(key)
        if val:
            resolved = resolve_media_reference(str(val), media_type="video", cache_dir="egolifeqa", env_vars=("EGOLIFEQA_VIDEO_DIR", "EGOLIFEQA_CACHE_DIR"))
            if isinstance(resolved, str) and os.path.exists(resolved):
                return [resolved]
            if isinstance(resolved, str):
                return [resolved]

    found = _find_nearest_video(participant, day, time_str)
    if found:
        return [found]

    # Try constructing HF hub path
    vid_name = f"{day}_{participant}_{str(time_str)[:8]}.mp4"
    # Attempt hf_hub_download style path (will be downloaded lazily by model if needed)
    # Return a path that media_resolver would resolve
    fallback = resolve_media_reference(vid_name, media_type="video", cache_dir="egolifeqa", env_vars=("EGOLIFEQA_VIDEO_DIR", "EGOLIFEQA_CACHE_DIR"))
    if isinstance(fallback, str):
        if os.path.exists(fallback):
            return [fallback]
        # If not exists, warn and return fallback path anyway (model may handle missing gracefully)
        eval_logger.warning(f"EgoLifeQA: video not found for {participant}/{day}/{time_str}, fallback {fallback}")
        return [fallback]
    eval_logger.warning(f"EgoLifeQA: no video found for {participant}/{day}/{time_str}")
    return []


def egolifeqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    question = str(doc.get("question", "")).strip()
    options = []
    # Support both choice_a/b/c/d and options list
    if "choice_a" in doc:
        for key in ("choice_a", "choice_b", "choice_c", "choice_d", "choice_e"):
            if key in doc and doc[key]:
                letter = chr(ord("A") + len(options))
                options.append(f"{letter}. {doc[key]}")
    elif "options" in doc:
        opts = doc["options"]
        if isinstance(opts, list):
            for i, opt in enumerate(opts):
                options.append(f"{chr(ord('A') + i)}. {opt}")
        elif isinstance(opts, dict):
            for letter in sorted(opts.keys()):
                options.append(f"{letter}. {opts[letter]}")
    elif "option" in doc:
        opts = doc["option"]
        if isinstance(opts, list):
            for op in opts:
                options.append(str(op))
        elif isinstance(opts, dict):
            for letter in sorted(opts.keys()):
                options.append(f"{letter}. {opts[letter]}")

    prompt = question
    if options:
        prompt = f"{question}\n" + "\n".join(options)

    return f"{pre_prompt}{prompt}{post_prompt}"


def egolifeqa_doc_to_choice(doc):
    choices = []
    if "choice_a" in doc:
        for key in ("choice_a", "choice_b", "choice_c", "choice_d", "choice_e"):
            if key in doc and doc[key] is not None:
                choices.append(str(doc[key]).strip())
        # filter empty
        choices = [c for c in choices if c]
        return choices
    if "options" in doc:
        opts = doc["options"]
        if isinstance(opts, list):
            return [str(o).strip() for o in opts]
        if isinstance(opts, dict):
            return [str(opts[k]).strip() for k in sorted(opts.keys())]
    if "option" in doc:
        opts = doc["option"]
        if isinstance(opts, list):
            # handle "A. xxx" format
            return [str(op).split(".", 1)[-1].strip() if "." in str(op) else str(op).strip() for op in opts]
        if isinstance(opts, dict):
            return [str(opts[k]).strip() for k in sorted(opts.keys())]
    # fallback: try choice_*
    for key in ("choice_a", "choice_b", "choice_c", "choice_d"):
        if doc.get(key):
            choices.append(str(doc[key]).strip())
    return choices


def egolifeqa_doc_to_target(doc):
    answer = str(doc.get("answer", "")).strip()
    # Normalize to upper letter
    if len(answer) == 1 and answer.upper() in "ABCDE":
        choices = egolifeqa_doc_to_choice(doc)
        idx = ord(answer.upper()) - ord("A")
        if 0 <= idx < len(choices):
            return choices[idx]
        return answer.upper()
    # If answer is full text, return as is if in choices
    choices = egolifeqa_doc_to_choice(doc)
    if answer in choices:
        return answer
    # Try case-insensitive match
    for c in choices:
        if c.lower() == answer.lower():
            return c
    return answer
