"""SciBench scoring: problems whose unit carries a power of ten are scored against the full value.

The documents below keep the format of the lmms-lab-encoder/SciBench_* datasets: the gold number is
stored in ``answer_number`` and the power of ten in ``unit`` (for example "$10^{11} \\mathrm{~J}$"
with ``answer_number`` "2.57" means 2.57 x 10^11 J).
"""

import pytest

from lmms_eval.tasks.scibench import utils

# (unit, answer_number, power of ten), taken from the physics split.
POWER_OF_TEN_PROBLEMS = [
    ("$10^{11} \\mathrm{~J}$", "2.57", "11"),
    ("$10^{-2} \\mathrm{~s}^{-1}$", "6.9", "-2"),
    ("$10^7 \\mathrm{yr}$", "9", "7"),
    ("$10^3 \\mathrm{~m} / \\mathrm{s}$", "2.16", "3"),
]


def score(unit, answer_number, response):
    return utils.scibench_process_results({"unit": unit, "answer_number": answer_number}, [response])["accuracy"]


@pytest.mark.parametrize("unit, answer_number, power", POWER_OF_TEN_PROBLEMS)
def test_exact_answer_in_scientific_notation_is_correct(unit, answer_number, power):
    # The gold value was cut to its first character ("2.57e11" became "2"), so this scored 0.
    assert score(unit, answer_number, f"The answer is therefore \\boxed{{{answer_number} \\times 10^{{{power}}}}}.") == 1


def test_exact_answer_written_out_is_correct():
    assert score("$10^{11} \\mathrm{~J}$", "2.57", "The answer is therefore \\boxed{257000000000}.") == 1


@pytest.mark.parametrize("unit, answer_number, power", POWER_OF_TEN_PROBLEMS)
def test_answer_within_tolerance_is_correct(unit, answer_number, power):
    # equiv_with_unit allows a 5% relative error.
    assert score(unit, answer_number, f"\\boxed{{{float(answer_number) * 1.02} \\times 10^{{{power}}}}}") == 1


@pytest.mark.parametrize("unit, answer_number, power", POWER_OF_TEN_PROBLEMS)
def test_wrong_exponent_is_incorrect(unit, answer_number, power):
    assert score(unit, answer_number, f"\\boxed{{{answer_number} \\times 10^{{{int(power) + 1}}}}}") == 0


@pytest.mark.parametrize(
    "unit, answer_number, response",
    [
        ("$10^{11} \\mathrm{~J}$", "2.57", "\\boxed{2}"),
        ("$10^{11} \\mathrm{~J}$", "2.57", "\\boxed{2.57}"),
        ("$10^{-2} \\mathrm{~s}^{-1}$", "6.9", "\\boxed{0}"),
        ("$10^7 \\mathrm{yr}$", "9", "\\boxed{9}"),
        ("$10^3 \\mathrm{~m} / \\mathrm{s}$", "2.16", "\\boxed{2}"),
    ],
)
def test_answer_without_the_power_of_ten_is_incorrect(unit, answer_number, response):
    # The truncated gold was only the first character of the value ("2", "0", "9"), so a bare "2" or "9" scored 1.
    assert score(unit, answer_number, response) == 0


def test_unconvertible_gold_number_is_compared_as_written():
    assert score("$10^{11} \\mathrm{~J}$", "n/a", "\\boxed{2.57 \\times 10^{11}}") == 0


@pytest.mark.parametrize(
    "unit, answer_number, response, expected",
    [
        ("$\\mathrm{~m}$", "2.57", "\\boxed{2.57}", 1),
        ("$\\mathrm{~m}$", "2.57", "\\boxed{2.6}", 1),
        ("$\\mathrm{~m}$", "2.57", "\\boxed{2}", 0),
        ("$\\mathrm{~m}$", "2.57", "\\boxed{257}", 0),
        ("$\\mathrm{~m} / \\mathrm{s}$", "2.16", "\\boxed{2.16}", 1),
        ("", "46.7", "\\boxed{46.7}", 1),
        ("", "46.7", "\\boxed{4}", 0),
    ],
)
def test_plain_units_are_scored_as_before(unit, answer_number, response, expected):
    assert score(unit, answer_number, response) == expected
