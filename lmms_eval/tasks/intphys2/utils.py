from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Final, Literal

from huggingface_hub import hf_hub_download

_REPO_ID: Final = "facebook/IntPhys2"
_DATASET_TYPE: Final = "dataset"
_POSSIBLE: Final = "possible"
_IMPOSSIBLE: Final = "impossible"

IntPhys2Label = Literal["possible", "impossible"]


def _video_path(doc: Mapping[str, str | None], split: str) -> str:
    filename = doc["file_name"]
    if filename is None:
        return ""
    return hf_hub_download(repo_id=_REPO_ID, repo_type=_DATASET_TYPE, filename=f"{split}/{filename}")


def intphys2_main_doc_to_visual(doc: Mapping[str, str | None]) -> list[str]:
    return [_video_path(doc, "Main")]


def intphys2_debug_doc_to_visual(doc: Mapping[str, str | None]) -> list[str]:
    return [_video_path(doc, "Debug")]


def intphys2_doc_to_text(doc: Mapping[str, str | None], lmms_eval_specific_kwargs: Mapping[str, str] | None = None) -> str:
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "\nAnswer with only Possible or Impossible.")
    condition = doc.get("condition") or "physical plausibility"
    camera = doc.get("Camera") or "unknown"
    difficulty = doc.get("Difficulty") or "unknown"
    return f"{pre_prompt}Is the event in this video physically possible or impossible? Condition: {condition}. Camera: {camera}. Difficulty: {difficulty}.{post_prompt}"


def intphys2_doc_to_target(doc: Mapping[str, str | None]) -> str:
    label = _target_label(doc)
    return "Possible" if label == _POSSIBLE else "Impossible"


def intphys2_process_results(doc: Mapping[str, str | None], results: Sequence[str]) -> dict[str, float]:
    prediction = _prediction_label(results[0] if results else "")
    target = _target_label(doc)
    return {"intphys2_accuracy": 1.0 if prediction == target else 0.0}


def intphys2_aggregate(results: Sequence[float]) -> float:
    return sum(results) / len(results) if results else 0.0


def _target_label(doc: Mapping[str, str | None]) -> IntPhys2Label:
    raw_label = (doc.get("type") or "").lower()
    if _IMPOSSIBLE in raw_label:
        return _IMPOSSIBLE
    return _POSSIBLE


def _prediction_label(text: str) -> IntPhys2Label | None:
    normalized = text.strip().lower()
    if re.search(r"\b(impossible|not possible|cannot|can't)\b", normalized):
        return _IMPOSSIBLE
    if re.search(r"\bpossible\b", normalized):
        return _POSSIBLE
    return None
