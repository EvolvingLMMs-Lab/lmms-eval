"""worldsense, av_odyssey and tomato share one copy of the MMMU option-letter parser (#1672)."""

import importlib

import pytest

INDEX2ANS = {"A": "a red car", "B": "a blue bus", "C": "a green truck", "D": "a bicycle"}
MODULES = ["worldsense", "av_odyssey", "tomato"]


@pytest.mark.parametrize("module", MODULES)
@pytest.mark.parametrize(
    "response",
    [
        "C",
        "C.",
        "(C)",
        "**Answer:** C",
        "Answer: C",
        "Based on the chart, the answer is C.",
        "C. a green truck",
        "(C) a green truck",
        "The answer is (C).",
        "Not (A) or (B), the answer is (C).",
    ],
)
def test_reads_the_answer_letter_not_capitals_in_words(module, response):
    parse = importlib.import_module(f"lmms_eval.tasks.{module}.utils").parse_multi_choice_response
    assert parse(response, ["A", "B", "C", "D"], INDEX2ANS) == "C"
