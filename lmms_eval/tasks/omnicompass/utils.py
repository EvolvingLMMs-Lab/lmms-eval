"""Structured messages and paired metrics for OmniCompass."""

from __future__ import annotations

import json
import os
from functools import lru_cache

from lmms_eval.tasks.omnicompass.prompts import render_prompt
from lmms_eval.tasks.omnicompass.protocol import Protocol, evaluate


@lru_cache(maxsize=1)
def protocol() -> Protocol:
    """One protocol instance per worker; media uses the canonical locked cache."""
    return Protocol.from_env()


def doc_to_text(doc: dict, lmms_eval_specific_kwargs=None) -> str:
    """Provide the text context used by the framework's request bookkeeping."""
    question = json.loads(doc["question_json"])
    duration = protocol().videos[question["video_id"]]["duration_seconds"]
    return render_prompt(question, doc["setting"], duration)


def doc_to_visual(doc: dict) -> list:
    """Reject legacy/simple backends that lose interleaved timestamp positions."""
    raise ValueError("OmniCompass requires a chat backend with doc_to_messages")


def doc_to_messages(doc: dict, lmms_eval_specific_kwargs=None) -> list[dict]:
    """Preserve frames, full audio, task reference and prompt in canonical order."""
    config = lmms_eval_specific_kwargs or {}
    supports_audio = config.get("supports_separate_audio", False)
    if config.get("api_audio_opt_in"):
        supports_audio = os.environ.get("OMNICOMPASS_OPENAI_AUDIO") == "input_audio"
    if doc["track"] == "AV" and not supports_audio:
        raise ValueError("AV requires a separate-audio chat backend. For a compatible API, set OMNICOMPASS_OPENAI_AUDIO=input_audio and model_args audio_format=input_audio.")
    content = [{"type": item["type"], "text" if item["type"] == "text" else "url": item["value"]} for item in protocol().content(doc)]
    return [{"role": "user", "content": content}]


def process_results(doc: dict, results: list) -> dict:
    """Keep raw outputs and pairing identities through distributed aggregation."""
    if len(results) != 1:
        raise ValueError("Exactly one final generation per document is required")
    row = {"doc": doc, "output": results[0]}
    return {key: row for key in ("text", "text_image", "cs")}


def aggregate(results: list, metric: str) -> float:
    """Reuse official scoring after gathering all ranks; reject incomplete pairs."""
    report = evaluate([r["doc"] for r in results], [r["output"] for r in results])
    if len(report["tracks"]) != 1:
        raise ValueError("Tracks must be aggregated separately")
    track = next(iter(report["tracks"].values()))
    tasks = {r["doc"]["task"] for r in results}
    if len(tasks) != 1:
        raise ValueError("Task metrics must be aggregated separately")
    task = tasks.pop()
    if metric == "cs":
        value = track["paired_cs"]["tasks"][task]["value"]
    else:
        setting = {"text": "text_target", "text_image": "reference_image"}[metric]
        key = {"TG": "TG_m_tIoU", "SG": "SG_m_sIoU", "QA": "QA_accuracy"}[task]
        value = track["main_metrics"].get(setting, {}).get(key)
    return float("nan") if value is None else float(value)


def aggregate_text(results: list) -> float:
    """Text target score on the official 0..1 scale."""
    return aggregate(results, "text")


def aggregate_text_image(results: list) -> float:
    """Text+Image target score on the official 0..1 scale."""
    return aggregate(results, "text_image")


def aggregate_cs(results: list) -> float:
    """Official paired CS, not an average of setting scores."""
    return aggregate(results, "cs")
