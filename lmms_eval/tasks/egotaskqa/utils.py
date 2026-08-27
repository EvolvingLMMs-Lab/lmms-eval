"""Utility functions for the EgoTaskQA-MCQ benchmark.

Annotations are loaded from the ``nv-njb/EgoTaskQA-MCQ`` HuggingFace dataset.
Videos are *not* redistributed — users must obtain them through the official
EgoTaskQA license process at https://sites.google.com/view/egotaskqa.

The video directory is resolved in this order:

1. ``EGOTASKQA_VIDEO_DIR`` environment variable, if set.
2. ``~/.cache/lmms_eval/egotaskqa/videos/`` (default).

Place the downloaded ``qa_videos/*.mp4`` files in that directory — no rename
is needed; filenames already match the ``video_path`` field.
"""

import os
import re
from pathlib import Path

from loguru import logger as eval_logger


def _video_dir() -> str:
    override = os.environ.get("EGOTASKQA_VIDEO_DIR")
    if override:
        return override
    return str(Path.home() / ".cache" / "lmms_eval" / "egotaskqa" / "videos")


def egotaskqa_doc_to_visual(doc):
    video_path = os.path.join(_video_dir(), doc["video_path"])
    if not os.path.exists(video_path):
        eval_logger.warning(f"Video not found: {video_path}. Set EGOTASKQA_VIDEO_DIR or place " f"qa_videos/ under ~/.cache/lmms_eval/egotaskqa/videos/.")
    return [video_path]


def egotaskqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    question = "Select the best answer to the following multiple-choice question " f"based on the video.\n{doc['q']}\nOptions:\n"
    options = doc["option"]
    for letter in sorted(options.keys()):
        question += f"({letter}) {options[letter]}\n"

    return f"{pre_prompt}{question}{post_prompt}"


def _extract_answer(response, options):
    letters = sorted(options.keys())

    response = response.replace("answer", "").replace("Answer", "")
    pred_answer = re.findall(r"[\(\ ]*([A-E])[\)\ ]*", response)

    if pred_answer:
        pred_letter = pred_answer[0].strip()
        if pred_letter in letters:
            return pred_letter

    for letter in letters:
        opt = options[letter].strip().strip(".")
        if opt.lower() in response.lower():
            return letter

    return ""


def egotaskqa_process_results(doc, results):
    pred = results[0]
    pred_ans = _extract_answer(pred, doc["option"])
    return {
        "egotaskqa_accuracy": {
            "pred_answer": pred_ans,
            "ground_truth": doc["a"],
        }
    }


def egotaskqa_aggregate_results(results):
    correct = sum(1 for r in results if r["pred_answer"] == r["ground_truth"])
    return correct / len(results) if results else 0
