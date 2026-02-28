import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from lmms_eval.tasks._task_utils.media_resolver import resolve_media_reference
from lmms_eval.utils import eval_logger

_VIDEO_EXTENSIONS = ("mp4", "MP4", "mkv", "webm", "mov")


def _normalize_text(text: Any) -> str:
    if text is None:
        return ""
    normalized = str(text).strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    if len(left) > len(right):
        left, right = right, left

    previous = list(range(len(left) + 1))
    for i, right_ch in enumerate(right, start=1):
        current = [i]
        for j, left_ch in enumerate(left, start=1):
            insertion = previous[j] + 1
            deletion = current[j - 1] + 1
            substitution = previous[j - 1] + (left_ch != right_ch)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def _anls_score(prediction: str, answer: str, threshold: float = 0.5) -> float:
    pred = _normalize_text(prediction)
    target = _normalize_text(answer)
    if not pred and not target:
        return 1.0
    if not pred or not target:
        return 0.0

    distance = _levenshtein_distance(pred, target)
    normalized_distance = distance / max(len(pred), len(target))
    score = 1.0 - normalized_distance
    if score < threshold:
        return 0.0
    return score


def _strip_answer_prefix(text: str) -> str:
    cleaned = str(text).strip()
    prefixes = [
        "the answer is",
        "answer:",
        "the correct answer is",
        "the final answer is",
    ]

    lowered = cleaned.lower()
    for prefix in prefixes:
        if lowered.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip(" :.-")
            break
    return cleaned


def _candidate_video_dirs() -> list[Path]:
    paths = []

    explicit_video_dir = os.getenv("EGOTEMPO_VIDEO_DIR", "").strip()
    if explicit_video_dir:
        paths.append(Path(os.path.expanduser(explicit_video_dir)))

    explicit_cache_dir = os.getenv("EGOTEMPO_CACHE_DIR", "").strip()
    if explicit_cache_dir:
        paths.append(Path(os.path.expanduser(explicit_cache_dir)))

    hf_home = Path(os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface/")))
    paths.append(hf_home / "egotempo")

    deduped = []
    seen = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _resolve_video_path(clip_id: str) -> str | None:
    if clip_id == "":
        return None

    resolved = resolve_media_reference(clip_id, media_type="video", cache_dir="egotempo", env_vars=("EGOTEMPO_VIDEO_DIR", "EGOTEMPO_CACHE_DIR"))
    if isinstance(resolved, str) and os.path.exists(resolved):
        return resolved

    for root in _candidate_video_dirs():
        for extension in _VIDEO_EXTENSIONS:
            candidate = root / f"{clip_id}.{extension}"
            if candidate.exists():
                return str(candidate)
    return None


def egotempo_doc_to_visual(doc):
    for key in ["video", "video_path", "media_path", "clip_path", "path", "file"]:
        value = doc.get(key)
        if value:
            resolved = resolve_media_reference(value, media_type="video", cache_dir="egotempo", env_vars=("EGOTEMPO_VIDEO_DIR", "EGOTEMPO_CACHE_DIR"))
            return [resolved]

    clip_id = str(doc.get("clip_id", "")).strip()
    video_path = _resolve_video_path(clip_id)
    if video_path is None:
        return []
    return [video_path]


def egotempo_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    question = str(doc.get("question", "")).strip()
    return f"{pre_prompt}{question}{post_prompt}"


def egotempo_doc_to_target(doc):
    return str(doc.get("answer", "")).strip()


def egotempo_process_results(doc, results):
    prediction = _strip_answer_prefix(results[0] if results else "")
    answer = str(doc.get("answer", "")).strip()
    score = _anls_score(prediction, answer)

    return {
        "egotempo_anls": {
            "score": score,
            "question_type": str(doc.get("question_type", "unknown")),
        }
    }


def egotempo_aggregate_results(items):
    if not items:
        return 0.0

    total_score = 0.0
    by_category = defaultdict(list)

    for item in items:
        score = float(item.get("score", 0.0))
        category = str(item.get("question_type", "unknown"))
        total_score += score
        by_category[category].append(score)

    for category in sorted(by_category):
        scores = by_category[category]
        category_score = sum(scores) / len(scores)
        eval_logger.info("EgoTempo [{}] ANLS: {:.2f}", category, category_score * 100)

    overall = total_score / len(items)
    eval_logger.info("EgoTempo overall ANLS: {:.2f}", overall * 100)
    return overall
