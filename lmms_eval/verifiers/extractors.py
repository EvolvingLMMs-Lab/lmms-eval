"""Per-sample text extractors for the verification pipeline.

Each extractor takes a single string and returns a cleaned/extracted string.
They complement the batch-level ``Filter`` system with lightweight callables
that compose naturally inside a ``VerificationPipeline``.
"""

import re
from typing import Any, List, Optional

from .base import Extractor


class StripReasoningExtractor(Extractor):
    """Strip ``<think>…</think>`` and similar reasoning blocks."""

    _DEFAULT_TAG_PAIRS = [
        ["<think>", "</think>"],
        ["<thinking>", "</thinking>"],
    ]

    def __init__(self, tag_pairs: Optional[List[List[str]]] = None):
        self.tag_pairs = tag_pairs or self._DEFAULT_TAG_PAIRS

    def extract(self, text: str, **kwargs: Any) -> str:
        if not isinstance(text, str):
            return str(text) if text is not None else ""
        from lmms_eval.api.reasoning import strip_reasoning_tags

        return strip_reasoning_tags(text, self.tag_pairs)


class RegexExtractor(Extractor):
    """Extract the first match of a regex pattern."""

    def __init__(self, pattern: str, group: int = 1, fallback: str = "", flags: int = 0):
        self.regex = re.compile(pattern, flags)
        self.group = group
        self.fallback = fallback

    def extract(self, text: str, **kwargs: Any) -> str:
        match = self.regex.search(text)
        if match:
            try:
                return match.group(self.group).strip()
            except IndexError:
                return match.group(0).strip()
        return self.fallback


class MCQExtractor(Extractor):
    """Extract a multiple-choice answer letter (A/B/C/…).

    Delegates to the shared ``extract_mcq_answer`` utility which handles
    10+ common answer formats with priority ranking.
    """

    def __init__(self, choices: Optional[List[str]] = None):
        self.choices = choices

    def extract(self, text: str, **kwargs: Any) -> str:
        from lmms_eval.tasks._task_utils.mcq_extract import extract_mcq_answer

        choices = kwargs.get("choices", self.choices)
        return extract_mcq_answer(text, choices)


class NumberExtractor(Extractor):
    r"""Extract a numerical answer.

    Priority order:
    1. ``\boxed{…}`` (LaTeX convention)
    2. Last number in the text (closest to the final answer)
    """

    _BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
    _NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")

    def __init__(self, fallback: str = ""):
        self.fallback = fallback

    def extract(self, text: str, **kwargs: Any) -> str:
        m = self._BOXED_RE.search(text)
        if m:
            return m.group(1).strip()
        numbers = self._NUMBER_RE.findall(text)
        if numbers:
            return numbers[-1].replace(",", "")
        return self.fallback


class StripExtractor(Extractor):
    """Strip whitespace and common model special tokens."""

    _SPECIAL_TOKENS = [
        "</s>",
        "<|endoftext|>",
        "<|end_of_sentence|>",
        "<|im_end|>",
        "[/INST]",
    ]

    def extract(self, text: str, **kwargs: Any) -> str:
        result = text
        for tok in self._SPECIAL_TOKENS:
            result = result.replace(tok, "")
        return result.strip()
