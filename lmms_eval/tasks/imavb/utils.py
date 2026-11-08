"""IMAVB: premise-perception conflict detection on long-form movie clips.

Each item is a six-way multiple-choice question about a movie clip (video with
audio). A-D are content answers; E ("The visual detail in the question is
incorrect") and F ("The audio detail in the question is incorrect") reject a
false premise. The four splits cross the target modality (vision, audio) with
the premise condition (standard, misleading).

Paper: https://arxiv.org/abs/2605.13737
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from lmms_eval.tasks._task_utils.mcq_extract import extract_mcq_answer
from lmms_eval.utils import resolve_cache_dir

OPTION_LETTERS = ["A", "B", "C", "D", "E", "F"]

with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    # yaml.safe_load cannot parse the !function tags, so drop those lines.
    _config = yaml.safe_load("".join(line for line in f if "!function" not in line))

_HF_HOME = os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface/"))
CACHE_DIR = resolve_cache_dir(_config["dataset_kwargs"]["cache_dir"], base_dir=_HF_HOME)


def imavb_doc_to_visual(doc: Dict[str, Any]) -> List[str]:
    """Return the clip path; the model reads both the frames and the audio track."""
    video_path = os.path.join(CACHE_DIR, doc["video"])
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"IMAVB video not found: {video_path}")
    return [video_path]


def imavb_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, str]] = None) -> str:
    """Build the A1 baseline prompt of the paper (fixed option order A-F)."""
    kwargs = lmms_eval_specific_kwargs or {}
    options = "\n".join(f"{letter}. {doc[f'option_{letter.lower()}']}" for letter in OPTION_LETTERS)
    prompt = f"Question:\n{doc['question']}\n\nOptions:\n{options}"
    return f"{kwargs.get('pre_prompt', '')}{prompt}{kwargs.get('post_prompt', '')}"


def _normalize(text: str) -> str:
    return " ".join(text.lower().split()).strip(" .")


def _match_option_text(response: str, doc: Dict[str, Any]) -> str:
    """Return the letter whose option text is the only one quoted in the response."""
    text = _normalize(response)
    hits = []
    for letter in OPTION_LETTERS:
        option = _normalize(str(doc[f"option_{letter.lower()}"]))
        if option and re.search(rf"(?<!\w){re.escape(option)}(?!\w)", text):
            hits.append(letter)
    return hits[0] if len(hits) == 1 else ""


def imavb_extract_answer(response: str, doc: Dict[str, Any]) -> str:
    """Extract an option letter A-F, or "" when the response is absent or ambiguous.

    The shared conservative letter extractor runs first. When it abstains, a
    response that quotes exactly one option text maps to that option.
    """
    letter = extract_mcq_answer(response, choices=OPTION_LETTERS)
    return letter or _match_option_text(response, doc)


def imavb_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, float]:
    response = results[0] if results else ""
    pred = imavb_extract_answer(response or "", doc)
    return {"accuracy": float(pred == doc["correct_answer"])}
