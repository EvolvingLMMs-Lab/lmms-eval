"""Robust multiple-choice answer extraction.

Shared utility for benchmark tasks that need to extract a choice letter
(A/B/C/D/...) from free-form model output.  Handles 10+ common answer
formats and uses a priority ranking to pick the best candidate.

Usage::

    from lmms_eval.tasks._task_utils.mcq_extract import extract_mcq_answer

    letter = extract_mcq_answer("The correct answer is (B).")  # -> "B"
"""

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
    # Markup-style answer tags
    "<answer",
    "answer>",
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

    # Match both cases of each choice so lowercase model output ("the answer
    # is b", "(b) ...") is recognized; candidates always carry the choice
    # letter exactly as given in ``choices``.
    case_variants = [(ch, letter) for ch in all_choices for letter in dict.fromkeys([ch, ch.lower()])]

    # --- (A) ---
    for ch, letter in case_variants:
        if f"({letter})" in text:
            candidates.append((ch, text.rfind(f"({letter})"), "parentheses"))

    # --- A. ---
    for ch, letter in case_variants:
        if f"{letter}." in text:
            candidates.append((ch, text.rfind(f"{letter}."), "period"))

    # --- A: ---
    for ch, letter in case_variants:
        if f"{letter}:" in text:
            candidates.append((ch, text.rfind(f"{letter}:"), "colon"))

    # --- A) ---
    for ch, letter in case_variants:
        if f"{letter})" in text:
            candidates.append((ch, text.rfind(f"{letter})"), "right_paren"))

    # --- A followed by space ---
    # Requires a real character before the letter: the padding space must not
    # count as a boundary, and the trailing "A " of an acronym ("DNA helix")
    # must not match.  Stays uppercase-only because lowercase " a " is the
    # English article and appears throughout ordinary prose.
    for ch, letter in case_variants:
        if not letter.isupper():
            continue
        pos = text.rfind(f"{letter} ")
        if pos >= 2 and not text[pos - 1].isalnum():
            candidates.append((ch, pos, "space"))

    # --- Common answer phrases ("the answer is A", etc.) ---
    text_lower = text.lower()
    # Prefer uppercase letters: trailing prose after the letter ("C and
    # stop") contains lowercase a-h words ("and") that must not win.  A
    # lowercase letter is only accepted when no uppercase one follows.
    for uppercase_only in (True, False):
        found_before = len(candidates)
        for phrase in _ANSWER_PHRASES:
            idx = text_lower.find(phrase)
            if idx != -1:
                after = idx + len(phrase)
                for ch, letter in case_variants:
                    if uppercase_only != letter.isupper():
                        continue
                    if uppercase_only:
                        ch_pos = text.find(letter, after)
                    else:
                        ch_pos = text_lower.find(letter.lower(), after)
                    if ch_pos != -1:
                        candidates.append((ch, ch_pos, "phrase"))
        if len(candidates) > found_before:
            break

    # --- Starts with standalone choice letter (not part of a word) ---
    stripped = text.strip()
    for ch, letter in case_variants:
        if not stripped.startswith(letter):
            continue
        rest = stripped[len(letter) :]
        first_word = rest.split()[0] if rest.split() else ""
        # A response-initial "A" followed by a lowercase word is the English
        # article ("A triangle has three sides"), not a choice; "B is
        # correct" style statements stay valid.
        if ch == "A" and first_word and first_word[0].islower() and first_word != "is":
            continue
        if len(rest) == 0 or not rest[0].isalpha():
            candidates.append((ch, 0, "start"))

    # --- Ends with standalone choice letter ---
    # Stays uppercase-only: prose routinely ends in a lowercase letter
    # ("the value of c", "plan b") that is not a choice.
    for ch, letter in case_variants:
        if not letter.isupper():
            continue
        if stripped.endswith(letter) and (len(stripped) == len(letter) or not stripped[-2].isalpha()):
            candidates.append((ch, len(text) - 1, "end"))

    # --- Fallback: any occurrence (lowest priority) ---
    if not candidates:
        for ch in all_choices:
            pos = text.rfind(ch)
            if pos == -1:
                continue
            # A letter embedded in a larger token ("DNA") is not an answer.
            if text[pos - 1].isalnum() or text[pos + 1].isalnum():
                continue
            # Same article guard as the start format, for a bare sentence-
            # initial "A" that no other format matched.
            if ch == "A" and pos == 1 and text[pos + 1 :].lstrip()[:1].islower():
                continue
            candidates.append((ch, pos, "fallback"))

    if not candidates:
        return ""

    # Sort by (priority DESC, position DESC) — highest-priority format
    # wins; within the same format, later position (closer to end) wins.
    candidates.sort(
        key=lambda x: (_FORMAT_PRIORITY.get(x[2], 0), x[1]),
        reverse=True,
    )
    return candidates[0][0]
