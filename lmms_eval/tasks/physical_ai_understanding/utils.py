"""Physical AI Understanding task for lmms-eval.

A video MCQ benchmark from NVIDIA's Cosmos PhysicalAI family covering
embodied/AV/robotics reasoning. The HF dataset bundles the QA parquet at
``data/test-*.parquet`` and the source videos at ``videos/<subset>/<id>.mp4``.

Each item carries a structured ``index2ans`` mapping ({"A": ..., "B": ...,
"C": ..., "D": ...}) so we don't need to re-parse choices from the prompt.

Reference dataset: https://huggingface.co/datasets/shi-labs/physical-ai-bench-understanding
"""

from __future__ import annotations

import os
import os.path as osp
import re
from functools import lru_cache
from typing import Any, Dict, List

from huggingface_hub import snapshot_download

from lmms_eval.filters.extraction import ExtendedRegexFilter
from lmms_eval.filters.transformation import MapFilter

REPO_ID = "shi-labs/physical-ai-bench-understanding"

REPLACE_PROMPT = "Please answer directly with only the letter of the correct option and nothing else."


@lru_cache(maxsize=1)
def _video_root() -> str:
    """Download (once) and return the local path to the dataset snapshot."""
    return snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=["videos/**"],
    )


def physical_ai_understanding_doc_to_visual(doc: Dict[str, Any]) -> List[str]:
    path = osp.join(_video_root(), doc["video_path"])
    if not osp.exists(path):
        raise FileNotFoundError(f"video not found: {path}")
    return [path]


def physical_ai_understanding_doc_to_text(
    doc: Dict[str, Any],
    lmms_eval_specific_kwargs: Dict[str, Any] | None = None,
) -> str:
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    question = doc["question"].strip()
    if post_prompt:
        question = question.replace(REPLACE_PROMPT, "")

    index2ans = doc.get("index2ans") or {}
    options = []
    for letter in sorted(index2ans.keys()):
        ans_text = index2ans[letter]
        if ans_text is not None:
            options.append(f"{letter}. {ans_text}")

    return f"{pre_prompt}{question}\n\n" + "\n".join(options) + f"\n{post_prompt}"


class NumberWordsToDigitsFilter(MapFilter):
    def __init__(self) -> None:
        mapping_dict = {
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
        }
        super().__init__(mapping_dict, default_value=None)

    def apply(self, resps, docs):
        def filter_set(inst):
            return [self.mapping_dict.get(resp.lower(), resp) for resp in inst]

        return [filter_set(resp) for resp in resps]


class MultiChoiceRegexFilter(ExtendedRegexFilter):
    """Letter-or-choice-text extractor for index2ans-style MCQ.

    Tries (1) a leading uppercase letter, then (2) match against any
    choice text from ``index2ans`` (falling back to parsing ``A. ...``
    style options out of the question text if ``index2ans`` is missing).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def apply(self, resps, docs):
        filtered_resps = []
        for r, doc in zip(resps, docs):
            fallback_regexes = []
            choice_to_alpha = {}

            index2ans = doc.get("index2ans") or {}
            if index2ans:
                for letter in sorted(index2ans.keys()):
                    ans_text = index2ans[letter]
                    if ans_text is not None:
                        choice_text = ans_text.strip()
                        fallback_regexes.append(re.escape(choice_text))
                        choice_to_alpha[choice_text] = letter
            else:
                # No structured choices — parse from question
                for m in re.finditer(r"\b([A-Z])\.\s+([^\n]*)", doc.get("question", "")):
                    choice_text = m.group(2).strip()
                    fallback_regexes.append(re.escape(choice_text))
                    choice_to_alpha[choice_text] = m.group(1)

            letter_regex = re.compile(r"^([A-Z])\b")

            filtered = []
            for resp in r:
                resp = re.sub(r"<think>.*?</think>", "", resp, flags=re.DOTALL).strip()
                resp = re.sub(r"<thought>.*?</thought>", "", resp, flags=re.DOTALL).strip()
                ans_match = re.search(r"<answer>(.*?)</answer>", resp, flags=re.DOTALL)
                if ans_match:
                    resp = ans_match.group(1).strip()
                cleaned = re.sub(r"[^\w\s]", "", resp).strip()

                letter_match = letter_regex.match(cleaned)
                if letter_match and letter_match.group(1) in index2ans:
                    filtered.append(letter_match.group(1))
                elif fallback_regexes:
                    fallback_regex = re.compile("|".join(fallback_regexes))
                    match = fallback_regex.search(cleaned)
                    if match and match.group() in choice_to_alpha:
                        filtered.append(choice_to_alpha[match.group()])
                    else:
                        filtered.append(cleaned)
                else:
                    filtered.append(cleaned)

            filtered_resps.append(filtered[0])

        return filtered_resps
