"""CRPE-Relation task utilities.

Annotations come from ``OpenGVLab/CRPE/crpe_relation.jsonl``. Image paths in
the ``image`` field have one of two prefixes:

- ``abnormal_images/*.jpg`` — fetched from the same CRPE HF dataset
  (via ``huggingface_hub.snapshot_download``).
- ``coco/val2017/*.jpg`` — MS-COCO val2017 images. These are *not*
  redistributed from CRPE; users must download them once:

  ```bash
  wget http://images.cocodataset.org/zips/val2017.zip
  unzip val2017.zip
  export COCO_VAL2017_DIR=/path/to/val2017
  ```

  (or place them at ``~/.cache/lmms_eval/coco/val2017/``).
"""

import os
import re
from pathlib import Path

from huggingface_hub import snapshot_download
from lmms_eval.filters.extraction import ExtendedRegexFilter
from lmms_eval.filters.transformation import MapFilter
from loguru import logger as eval_logger

REPLACE_PROMPT = (
    "Please answer directly with only the letter of the correct option and nothing else."
)

_CRPE_ROOT = None


def _crpe_root() -> str:
    """Return the local snapshot of OpenGVLab/CRPE (for abnormal_images/)."""
    global _CRPE_ROOT
    if _CRPE_ROOT is None:
        _CRPE_ROOT = snapshot_download(
            repo_id="OpenGVLab/CRPE",
            repo_type="dataset",
            allow_patterns=["abnormal_images/*"],
        )
    return _CRPE_ROOT


def _coco_val2017_dir() -> str:
    override = os.environ.get("COCO_VAL2017_DIR")
    if override:
        return override
    return str(Path.home() / ".cache" / "lmms_eval" / "coco" / "val2017")


def crpe_relation_doc_to_visual(doc):
    rel = doc["image"]  # e.g. "coco/val2017/000000xxx.jpg" or "abnormal_images/123.jpg"
    if rel.startswith("coco/val2017/"):
        leaf = rel.split("coco/val2017/", 1)[1]
        path = os.path.join(_coco_val2017_dir(), leaf)
    elif rel.startswith("abnormal_images/"):
        path = os.path.join(_crpe_root(), rel)
    else:
        path = rel
    if not os.path.exists(path):
        eval_logger.warning(
            f"Image not found: {path}. For COCO val2017, set COCO_VAL2017_DIR "
            "(see crpe_relation/utils.py docstring)."
        )
    return [path]


def crpe_relation_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    question = doc["text"].strip()
    if post_prompt:
        question = question.replace(REPLACE_PROMPT, "")
    return f"{pre_prompt}{question}\n{post_prompt}"


class NumberWordsToDigitsFilter(MapFilter):
    def __init__(self) -> None:
        super().__init__(
            {
                "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
                "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
                "ten": "10",
            },
            default_value=None,
        )

    def apply(self, resps, docs):
        return [[self.mapping_dict.get(r.lower(), r) for r in inst] for inst in resps]


class MultiChoiceRegexFilter(ExtendedRegexFilter):
    """Falls back to matching the choice text against the response when the
    primary regex doesn't match."""

    def apply(self, resps, docs):
        filtered_resps = []
        for r, doc in zip(resps, docs):
            fallback_regexes = []
            choice_to_alpha = {}
            next_alpha = "A"
            for m in re.compile(r"\b([A-Z])\.\s+([^\n]*)").findall(doc["text"]):
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
            filtered_resps.append(filtered[0])
        return filtered_resps
