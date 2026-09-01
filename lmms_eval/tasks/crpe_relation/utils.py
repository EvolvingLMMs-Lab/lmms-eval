"""CRPE-Relation task for lmms-eval.

Single-image MCQ on object/predicate/subject relationships. The bundled
re-host at ``nv-njb/CRPE`` ships annotations + images as an Image()
feature in a single parquet, so we just unpack the PIL image and feed
the existing question text (which already includes A./B./C./D. options).

Reference (annotations): https://huggingface.co/datasets/OpenGVLab/CRPE
Re-host (bundled images):  https://huggingface.co/datasets/nv-njb/CRPE
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from PIL import Image

from lmms_eval.filters.extraction import ExtendedRegexFilter
from lmms_eval.filters.transformation import MapFilter

REPLACE_PROMPT = "Please answer directly with only the letter of the correct option and nothing else."


def crpe_relation_doc_to_visual(doc: Dict[str, Any]) -> List[Image.Image]:
    return [doc["image"].convert("RGB")]


def crpe_relation_doc_to_text(
    doc: Dict[str, Any],
    lmms_eval_specific_kwargs: Dict[str, Any] | None = None,
) -> str:
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    question = doc["text"].strip()
    if post_prompt:
        question = question.replace(REPLACE_PROMPT, "")
    return f"{pre_prompt}{question}\n{post_prompt}"


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
    """Letter-or-choice-text extractor.

    The question text already contains ``A./B./C./D.`` options inline; we
    parse those once per doc and try (1) a leading uppercase letter, then
    (2) substring match against any of the choice texts.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def apply(self, resps, docs):
        filtered_resps = []
        for r, doc in zip(resps, docs):
            fallback_regexes = []
            choice_to_alpha = {}

            for m in re.finditer(r"\b([A-Z])\.\s+([^\n]*)", doc.get("text", "")):
                choice_text = m.group(2).strip()
                fallback_regexes.append(re.escape(choice_text))
                choice_to_alpha[choice_text] = m.group(1)

            fallback_regex = re.compile("|".join(fallback_regexes)) if fallback_regexes else None

            filtered = []
            for resp in r:
                # Strip common reasoning wrappers
                resp = re.sub(r"<think>.*?</think>", "", resp, flags=re.DOTALL).strip()
                resp = re.sub(r"<thought>.*?</thought>", "", resp, flags=re.DOTALL).strip()
                ans_match = re.search(r"<answer>(.*?)</answer>", resp, flags=re.DOTALL)
                if ans_match:
                    resp = ans_match.group(1).strip()
                cleaned = re.sub(r"[^\w\s]", "", resp).strip()

                if fallback_regex is not None:
                    match = fallback_regex.search(cleaned)
                    if match and match.group() in choice_to_alpha:
                        filtered.append(choice_to_alpha[match.group()])
                        continue
                filtered.append(cleaned)

            filtered_resps.append(filtered[0])

        return filtered_resps
