from __future__ import annotations

import json
import os
import re
from functools import lru_cache

from huggingface_hub import hf_hub_download
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

REPO_ID = "nvidia/PhysicalAI-VANTAGE-Bench"
CHOICES = ("A", "B", "C", "D")
SUBMISSION_TASK = "video_qa"


def _video_filename(q_uid: str) -> str:
    video = str(q_uid).strip()
    if video.endswith(".json"):
        video = video.removesuffix(".json")
    if not video.endswith(".mp4"):
        video = f"{video}.mp4"
    return video


def _canonical_id(video: str, index: int) -> str:
    return f"{video.removesuffix('.mp4')}__q_{index:06d}"


def _format_option(label: str, option: str) -> str:
    text = str(option).strip()
    match = re.match(r"^[A-D]\s*[:.)]\s*(.*)$", text)
    if match:
        text = match.group(1).strip()
    return f"{label}. {text}"


@lru_cache(maxsize=None)
def _download_video(video: str) -> str:
    return hf_hub_download(
        repo_id=REPO_ID,
        filename=f"data/vqa/videos/{video}",
        repo_type="dataset",
        token=os.getenv("HF_TOKEN") or None,
    )


def vantage_vqa_process_docs(dataset):
    def add_index_and_video(doc: dict, index: int) -> dict:
        doc["index"] = index
        doc["video"] = _video_filename(doc["q_uid"])
        return doc

    return dataset.map(add_index_and_video, with_indices=True)


def vantage_vqa_doc_to_visual(doc: dict) -> list[str]:
    return [_download_video(_video_filename(doc["q_uid"]))]


def vantage_vqa_doc_to_text(doc: dict, lmms_eval_specific_kwargs=None) -> str:
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    options = "\n".join(_format_option(label, option) for label, option in zip(CHOICES, doc["options"], strict=False))
    question = str(doc["question"]).strip()

    return f"{pre_prompt}Question: {question}\nChoices:\n{options}{post_prompt}"


def vantage_vqa_doc_to_messages(doc: dict, lmms_eval_specific_kwargs=None) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "video", "url": vantage_vqa_doc_to_visual(doc)[0]},
                {"type": "text", "text": vantage_vqa_doc_to_text(doc, lmms_eval_specific_kwargs)},
            ],
        },
    ]


def vantage_vqa_doc_to_target(_doc: dict) -> str:
    return ""


def vantage_vqa_process_results(doc: dict, results: list[str]) -> dict:
    prediction = results[0].strip()
    video = doc.get("video") or _video_filename(doc["q_uid"])
    index = int(doc["index"])
    return {
        "submission": {
            "id": _canonical_id(video, index),
            "task": SUBMISSION_TASK,
            "conversations": [{"from": "assistant", "value": prediction}],
            "metadata": {"model": "", "extra": {}},
        },
    }


def vantage_vqa_aggregate_submissions(results: list[dict], args) -> None:
    path = generate_submission_file("vantage_vqa_submission.jsonl", args)
    model_name = str(getattr(args, "model", "") or "")
    with open(path, "w", encoding="utf-8") as file:
        for result in results:
            metadata = dict(result.get("metadata", {}))
            metadata["model"] = model_name
            metadata.setdefault("extra", {})
            record = {**result, "metadata": metadata}
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    eval_logger.info("VANTAGE-VQA submission saved to {}", path)
