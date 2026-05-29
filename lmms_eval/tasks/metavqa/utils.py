"""MetaVQA-MCQ task utilities.

Annotations come from ``nv-njb/MetaVQA-MCQ`` (JSONL flattened from
``Weizhen011210/MetaVQA-Eval/test.json``). Images come from
``Weizhen011210/MetaVQA-Eval`` under ``obs/*.png`` and are fetched on first
use via ``huggingface_hub.snapshot_download``.

To skip the download (e.g. when an offline cache is already present), set
``METAVQA_IMAGE_DIR`` to the directory containing the ``obs/`` subdir.
"""

import os
import re

from huggingface_hub import snapshot_download

from lmms_eval.filters.extraction import ExtendedRegexFilter
from lmms_eval.filters.transformation import MapFilter

REPLACE_PROMPT = "Please answer directly with only the letter of the correct option and nothing else."

_IMAGE_ROOT = None


def _image_root() -> str:
    """Return the directory that contains the ``obs/`` subdir."""
    global _IMAGE_ROOT
    if _IMAGE_ROOT is not None:
        return _IMAGE_ROOT
    override = os.environ.get("METAVQA_IMAGE_DIR")
    if override:
        _IMAGE_ROOT = override
        return _IMAGE_ROOT
    _IMAGE_ROOT = snapshot_download(
        repo_id="Weizhen011210/MetaVQA-Eval",
        repo_type="dataset",
        allow_patterns=["obs/*"],
    )
    return _IMAGE_ROOT


def metavqa_doc_to_visual(doc):
    root = _image_root()
    paths = [os.path.join(root, p) for p in doc["obs"]]
    return [p for p in paths if os.path.exists(p)]


def metavqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    question = doc["question"].strip()
    if post_prompt:
        question = question.replace(REPLACE_PROMPT, "")
    return f"{pre_prompt}{question}\n{post_prompt}"


class NumberWordsToDigitsFilter(MapFilter):
    def __init__(self) -> None:
        super().__init__(
            {
                "zero": "0",
                "one": "1",
                "two": "2",
                "three": "3",
                "four": "4",
                "five": "5",
                "six": "6",
                "seven": "7",
                "eight": "8",
                "nine": "9",
                "ten": "10",
            },
            default_value=None,
        )

    def apply(self, resps, docs):
        return [[self.mapping_dict.get(r.lower(), r) for r in inst] for inst in resps]


class MultiChoiceRegexFilter(ExtendedRegexFilter):
    """Falls back to matching the choice text against the response when the
    primary regex (e.g. ``(\\([A-Z]\\))``) doesn't match."""

    def apply(self, resps, docs):
        filtered_resps = []
        for r, doc in zip(resps, docs):
            fallback_regexes = []
            choice_to_alpha = {}
            next_alpha = "A"
            for m in re.compile(r"\b([A-Z])\.\s+([^\n]*)").findall(doc["question"]):
                choice_text = m[1].strip()
                fallback_regexes.append(re.escape(choice_text))
                choice_to_alpha[choice_text] = next_alpha
                next_alpha = chr(ord(next_alpha) + 1)
            fallback_regex = re.compile("|".join(fallback_regexes)) if fallback_regexes else None

            filtered = []
            for resp in r:
                cleaned = re.sub(r"[^\w\s]", "", resp).strip()
                if fallback_regex:
                    match = fallback_regex.search(cleaned)
                    if match and match.group() in choice_to_alpha:
                        filtered.append(choice_to_alpha[match.group()])
                        continue
                filtered.append(cleaned)
            filtered_resps.append(filtered[0].split(" ")[0])
        return filtered_resps
