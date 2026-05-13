"""Robust multiple-choice answer extraction.

Shared utility for benchmark tasks that need to extract a choice letter
(A/B/C/D/...) from free-form model output.  Handles 10+ common answer
formats and uses a priority ranking to pick the best candidate.

Usage::

    from lmms_eval.tasks._task_utils.mcq_extract import extract_mcq_answer

    letter = extract_mcq_answer("The correct answer is (B).")  # -> "B"
"""

import re
from typing import List, Optional

_DEFAULT_CHOICES = ["A", "B", "C", "D", "E", "F", "G", "H"]

_ANSWER_PHRASES = [
    "the answer is",
    "answer is",
    "the correct answer is",
    "correct answer is",
    "the best answer is",
    "best answer is",
    "the correct option is",
    "correct option is",
    "the best option is",
    "best option is",
    "the choice is",
    "choice is",
    "the correct choice is",
    "correct choice is",
    "i choose",
    "i select",
    "i pick",
    "my answer is",
    "my choice is",
    # Korean
    "옵션",
    "정답은",
    "답은",
    "답:",
    # Chinese
    "答案是",
    "答案为",
    "选",
    # Japanese
    "答えは",
]

# Higher = more confident that this is the intended answer.
_FORMAT_PRIORITY = {
    "start": 10,
    "end": 9,
    "phrase": 7,
    "parentheses": 6,
    "period": 5,
    "colon": 4,
    "right_paren": 3,
    "space": 2,
    "fallback": 0,
}


def extract_mcq_answer(response: str, choices: Optional[List[str]] = None) -> str:
    """Extract a multiple-choice answer letter from model output.

    Searches for choice letters in various common formats and returns the
    best candidate using a priority ranking.  When multiple candidates
    match, prefers the **last** occurrence in the **highest-priority**
    format — this naturally handles reasoning-style outputs where the
    model discusses options before giving its final answer.

    Args:
        response: Model output (should already have ``<think>`` tags
            stripped by the postprocessing pipeline).
        choices: Valid choice letters.  Defaults to ``["A".."H"]``.

    Returns:
        Uppercase choice letter, or ``""`` if none found.
    """
    if not response or not response.strip():
        return ""

    all_choices = choices or _DEFAULT_CHOICES

    text = response.strip()
    for char in [",", ".", "!", "?", ";", ":", "'", '"']:
        text = text.strip(char)
    # Pad with spaces for boundary matching.
    text = " " + text + " "

    candidates: list = []  # (letter, position, format_name)

    # --- (A) ---
    for ch in all_choices:
        if f"({ch})" in text:
            candidates.append((ch, text.rfind(f"({ch})"), "parentheses"))

    # --- A. ---
    for ch in all_choices:
        if f"{ch}." in text:
            candidates.append((ch, text.rfind(f"{ch}."), "period"))

    # --- A: ---
    for ch in all_choices:
        if f"{ch}:" in text:
            candidates.append((ch, text.rfind(f"{ch}:"), "colon"))

    # --- A) ---
    for ch in all_choices:
        if f"{ch})" in text:
            candidates.append((ch, text.rfind(f"{ch})"), "right_paren"))

    # --- A followed by space ---
    for ch in all_choices:
        if f"{ch} " in text:
            candidates.append((ch, text.rfind(f"{ch} "), "space"))

    # --- Common answer phrases ("the answer is A", etc.) ---
    text_lower = text.lower()
    for phrase in _ANSWER_PHRASES:
        idx = text_lower.find(phrase)
        if idx != -1:
            after = idx + len(phrase)
            for ch in all_choices:
                ch_pos = text.find(ch, after)
                if ch_pos != -1:
                    candidates.append((ch, ch_pos, "phrase"))

    # --- Starts with standalone choice letter (not part of a word) ---
    stripped = text.strip()
    for ch in all_choices:
        if stripped.startswith(ch) and (len(stripped) == 1 or not stripped[1].isalpha()):
            candidates.append((ch, 0, "start"))

    # --- Ends with standalone choice letter ---
    for ch in all_choices:
        if stripped.endswith(ch) and (len(stripped) == 1 or not stripped[-2].isalpha()):
            candidates.append((ch, len(text) - 1, "end"))

    # --- Fallback: any occurrence (lowest priority) ---
    if not candidates:
        for ch in all_choices:
            if ch in text:
                candidates.append((ch, text.rfind(ch), "fallback"))

    if not candidates:
        return ""

    # Sort by (priority DESC, position DESC) — highest-priority format
    # wins; within the same format, later position (closer to end) wins.
    candidates.sort(
        key=lambda x: (_FORMAT_PRIORITY.get(x[2], 0), x[1]),
        reverse=True,
    )
    return candidates[0][0]
