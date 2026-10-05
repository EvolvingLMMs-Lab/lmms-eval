"""Reviewed data-driven contract for shared MCQ answer extraction."""

import json
from pathlib import Path
from typing import Any

import pytest

from lmms_eval.tasks._task_utils.mcq_extract import extract_mcq_answer
from lmms_eval.verifiers.extractors import MCQExtractor

_CORPUS_PATH = Path(__file__).with_name("mcq_extract") / "cases.json"
_CORPUS = json.loads(_CORPUS_PATH.read_text(encoding="utf-8"))
_CASES = _CORPUS["cases"]


def test_corpus_structure() -> None:
    """Fail loudly for duplicate IDs, malformed inputs, or unoffered outputs."""
    assert _CORPUS["schema_version"] == 1
    assert _CASES
    ids = [case["id"] for case in _CASES]
    assert len(ids) == len(set(ids))
    required = {"id", "category", "response", "choices", "expected", "rationale"}
    for case in _CASES:
        assert required <= case.keys(), case
        assert all(isinstance(case[key], str) and case[key].strip() for key in ("id", "category", "rationale")), case
        assert case["response"] is None or isinstance(case["response"], str), case
        choices = case["choices"]
        assert choices is None or isinstance(choices, list), case
        assert choices is None or all(isinstance(choice, str) and len(choice) == 1 and choice.isascii() and choice.isalpha() for choice in choices), case
        allowed = {choice.upper() for choice in choices} if choices else set("ABCDEFGH")
        assert isinstance(case["expected"], str), case
        assert case["expected"] == "" or case["expected"] in allowed, case
        if "base_expected" in case:
            assert isinstance(case["base_expected"], str), case
    assert {case["category"] for case in _CASES} >= {"regression", "legacy", "container", "phrase", "multilingual", "correction", "ambiguous", "boundary", "custom", "article", "model_reply"}
    assert sum(case["category"] == "regression" for case in _CASES) == 6
    assert sum(case["category"] == "legacy" for case in _CASES) == 26
    alphabet_sizes = {len(case["choices"] or list("ABCDEFGH")) for case in _CASES}
    assert alphabet_sizes >= {2, 3, 4, 5, 8, 14}


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case["id"])
def test_shared_extractor_corpus(case: dict[str, Any]) -> None:
    assert extract_mcq_answer(case["response"], choices=case["choices"]) == case["expected"], case["rationale"]


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case["id"])
def test_verifier_extractor_corpus(case: dict[str, Any]) -> None:
    assert MCQExtractor(choices=case["choices"]).extract(case["response"]) == case["expected"], case["rationale"]


# This small declared grammar is independent of the parser's implementation.
# Every offered letter is substituted, preventing a fix tuned only to A-D.
_ALPHABETS = ["AB", "ABC", "ABCD", "ABCDE", "ABCDEFGH", "ABCDEFGHIJKLMN", "ACF", "xz"]
_FORMATS = [
    "{letter}",
    "({letter})",
    "{letter}. Selected option.",
    "{letter}: Selected option.",
    "{letter}) Selected option.",
    "Answer: {letter}",
    "The answer is {letter} because the objects match.",
    "I choose {letter}.",
    "Option {letter} is correct.",
    "<answer>{letter}</answer>",
    r"\boxed{{{letter}}}",
    "**{letter}**",
    "`{letter}`",
]
_SUBSTITUTIONS = [(alphabet, letter.upper(), lower) for alphabet in _ALPHABETS for letter in alphabet for lower in (False, True)]


@pytest.mark.parametrize("template", _FORMATS)
@pytest.mark.parametrize("alphabet, expected, lower", _SUBSTITUTIONS, ids=[f"{alphabet}-{letter}-{'lower' if lower else 'upper'}" for alphabet, letter, lower in _SUBSTITUTIONS])
def test_letter_substitution(alphabet: str, expected: str, lower: bool, template: str) -> None:
    """A documented selection format must work for every offered ASCII letter."""
    letter = expected.lower() if lower else expected
    response = template.format(letter=letter)
    choices = list(alphabet)
    assert extract_mcq_answer(response, choices=choices) == expected
    assert MCQExtractor(choices=choices)(response) == expected


@pytest.mark.parametrize("template", ["Answer: {letter}", "<answer>{letter}</answer>", r"\boxed{{{letter}}}"])
@pytest.mark.parametrize("alphabet", _ALPHABETS)
def test_removing_selected_choice_causes_abstention(alphabet: str, template: str) -> None:
    """Removing the selected option must never select a surviving distractor."""
    selected = alphabet[-1]
    choices = list(alphabet[:-1])
    response = "A is printed in the image. " + template.format(letter=selected)
    assert extract_mcq_answer(response, choices=choices) == ""


@pytest.mark.parametrize("prefix", ["word", "_", "2", "é", "猫", "α"])
@pytest.mark.parametrize("suffix", [".", ":", ")"])
@pytest.mark.parametrize("letter", ["b", "C", "n", "Z"])
def test_label_punctuation_cannot_split_a_word(prefix: str, suffix: str, letter: str) -> None:
    """Punctuation after a word, identifier or Unicode token cannot create a choice."""
    response = f"{prefix}{letter}{suffix} This is descriptive text."
    assert extract_mcq_answer(response, choices=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")) == ""


@pytest.mark.parametrize("template", ["Answer: {letter}", "<answer>{letter}</answer>", r"\boxed{{{letter}}}"])
@pytest.mark.parametrize("letter", ["A", "B", "H", "I", "N", "Z"])
def test_explanation_cannot_replace_explicit_answer(letter: str, template: str) -> None:
    """Adding letter-heavy prose preserves the selected answer."""
    response = template.format(letter=letter.lower())
    explanation = " because a cubic object and DNA diagram appear beside an ID_B label."
    assert extract_mcq_answer(response + explanation, choices=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")) == letter
