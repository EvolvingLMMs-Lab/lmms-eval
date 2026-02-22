import os
import re
from pathlib import Path

import yaml

_CHOICE_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_DATASET_NAMES = [
    "action_antonym",
    "action_count",
    "action_localization",
    "action_sequence",
    "egocentric_sequence",
    "moving_direction",
    "object_count",
    "object_shuffle",
    "scene_transition",
    "unexpected_action",
]


def _safe_get(doc, keys, default=""):
    for key in keys:
        value = doc.get(key)
        if value is not None:
            return value
    return default


def _normalize_text(text):
    return " ".join(str(text or "").strip().lower().split())


def _extract_candidates(doc):
    candidates = doc.get("candidates", doc.get("options"))
    if isinstance(candidates, list):
        return [str(candidate) for candidate in candidates]

    options = []
    for index in range(len(_CHOICE_LETTERS)):
        option_key = f"option{index}"
        if option_key in doc and doc[option_key] not in (None, ""):
            options.append(str(doc[option_key]))
    return options


def _resolve_cache_dir():
    hf_home = os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface"))
    template_path = Path(__file__).parent / "_default_template_yaml"
    with open(template_path, "r", encoding="utf-8") as handle:
        raw = [line for line in handle.readlines() if "!function" not in line]
    config = yaml.safe_load("".join(raw)) or {}
    cache_name = config.get("dataset_kwargs", {}).get("cache_dir", "")
    if not cache_name:
        return None
    return os.path.join(hf_home, str(cache_name))


def _candidate_video_paths(video_name):
    if not _CACHE_DIR:
        return [video_name]

    relative_paths = [video_name, os.path.join("video", video_name), os.path.join("videos", video_name), os.path.join("data", video_name)]
    for dataset_name in _DATASET_NAMES:
        relative_paths.extend(
            [
                os.path.join(dataset_name, video_name),
                os.path.join("video", dataset_name, video_name),
                os.path.join("videos", dataset_name, video_name),
            ]
        )

    candidates = []
    for rel_path in relative_paths:
        abs_path = os.path.join(_CACHE_DIR, rel_path)
        if abs_path not in candidates:
            candidates.append(abs_path)
    return candidates


def _extract_choice_letter(prediction, candidates):
    text = str(prediction or "").strip()
    if not text:
        return ""

    all_choices = _CHOICE_LETTERS[: max(len(candidates), 2)]
    uppercase = text.upper()

    letter_match = re.search(r"\b([A-Z])\b", uppercase)
    if letter_match and letter_match.group(1) in all_choices:
        return letter_match.group(1)

    prefix_match = re.match(r"^\s*[\(\[]?([A-Z])[\)\].:]?", uppercase)
    if prefix_match and prefix_match.group(1) in all_choices:
        return prefix_match.group(1)

    normalized_pred = _normalize_text(text)
    matched_indices = []
    for index, candidate in enumerate(candidates):
        normalized_candidate = _normalize_text(candidate)
        if normalized_candidate and normalized_candidate in normalized_pred:
            matched_indices.append(index)
    if len(matched_indices) == 1:
        return all_choices[matched_indices[0]]

    return ""


def tvbench_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    video_value = _safe_get(doc, ["video", "video_path", "video_file"], "")
    if isinstance(video_value, dict):
        video_value = _safe_get(video_value, ["path", "video", "filename"], "")

    if isinstance(video_value, list):
        return [str(video) for video in video_value]

    video_name = str(video_value).strip()
    if not video_name:
        return []

    if os.path.isabs(video_name) and os.path.exists(video_name):
        return [video_name]

    for candidate in _candidate_video_paths(video_name):
        if os.path.exists(candidate):
            return [candidate]

    fallback_candidates = _candidate_video_paths(video_name)
    if fallback_candidates:
        return [fallback_candidates[0]]
    return [video_name]


def tvbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "Answer with the option letter only.")

    question = str(_safe_get(doc, ["question", "prompt", "query"], "")).strip()
    candidates = _extract_candidates(doc)

    lines = []
    if question:
        lines.append(question)
    for index, candidate in enumerate(candidates):
        lines.append(f"{_CHOICE_LETTERS[index]}. {candidate}")
    if post_prompt:
        lines.append(str(post_prompt).strip())

    text = "\n".join(lines).strip()
    if pre_prompt:
        text = f"{pre_prompt}{text}"
    return text


def tvbench_doc_to_target(doc, model_specific_target_kwargs=None):
    candidates = _extract_candidates(doc)
    answer = _safe_get(doc, ["answer", "correct_answer", "label", "correct_choice"], "")

    if isinstance(answer, int):
        if 0 <= answer < len(candidates):
            return _CHOICE_LETTERS[answer]
        if 1 <= answer <= len(candidates):
            return _CHOICE_LETTERS[answer - 1]

    text = str(answer).strip()
    if len(text) == 1 and text.isalpha():
        return text.upper()

    if text.isdigit():
        index = int(text)
        if 0 <= index < len(candidates):
            return _CHOICE_LETTERS[index]
        if 1 <= index <= len(candidates):
            return _CHOICE_LETTERS[index - 1]

    normalized_answer = _normalize_text(text)
    for index, candidate in enumerate(candidates):
        if _normalize_text(candidate) == normalized_answer:
            return _CHOICE_LETTERS[index]

    return text.upper()


def tvbench_process_results(doc, results):
    candidates = _extract_candidates(doc)
    prediction = results[0] if results else ""
    predicted_letter = _extract_choice_letter(prediction, candidates)
    target_letter = tvbench_doc_to_target(doc)
    return {"tvbench_acc": 1.0 if predicted_letter == target_letter else 0.0}


_CACHE_DIR = _resolve_cache_dir()
