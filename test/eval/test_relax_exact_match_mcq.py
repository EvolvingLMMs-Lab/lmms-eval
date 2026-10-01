"""Regression tests for MCQ detection in relax_exact_match.

parse_mcq extracts option letters from free text ("Africa" -> "A"), so it must
not be used to decide whether a ground truth is multiple choice. See #1546.
"""

from lmms_eval.tasks._task_utils.reasoning_utils import (
    is_mcq_ground_truth,
    relax_exact_match,
)


class TestIsMcqGroundTruth:
    def test_bare_letter_is_mcq(self):
        for letter in ["A", "B", "C", "D", "E", "F", "G", "H"]:
            assert is_mcq_ground_truth(letter)

    def test_wrapped_letter_is_mcq(self):
        assert is_mcq_ground_truth("(A)")
        assert is_mcq_ground_truth(" B. ")
        assert is_mcq_ground_truth("( D )")

    def test_free_text_is_not_mcq(self):
        assert not is_mcq_ground_truth("Africa")
        assert not is_mcq_ground_truth("Democrat (scores 60 to 100)")
        assert not is_mcq_ground_truth("Based on how much we can afford as a family")
        assert not is_mcq_ground_truth("12")
        assert not is_mcq_ground_truth("")

    def test_lowercase_letter_is_not_mcq(self):
        assert not is_mcq_ground_truth("a")


class TestRelaxExactMatchFreeTextGolds:
    def test_wrong_answer_sharing_initial_scores_zero(self):
        assert relax_exact_match("Asia", "Africa") == 0.0
        assert relax_exact_match("Belgium", "Brazil") == 0.0
        assert relax_exact_match("Canada", "China") == 0.0
        assert relax_exact_match("Avocado", "Apple") == 0.0

    def test_verbose_prediction_with_trailing_letter_scores_zero(self):
        assert relax_exact_match("I think the value is around 40 for D", "Democrats") == 0.0

    def test_matching_and_non_matching_controls(self):
        assert relax_exact_match("Africa", "Africa") == 1.0
        assert relax_exact_match("Europe", "Africa") == 0.0
        assert relax_exact_match("Kenya", "Nigeria") == 0.0
        assert relax_exact_match("13", "12") == 0.0

    def test_case_insensitive_exact_match(self):
        assert relax_exact_match("africa", "Africa") == 1.0
        assert relax_exact_match("AFRICA", "Africa") == 1.0


class TestRelaxExactMatchMcqGolds:
    def test_single_letter_gold_keeps_mcq_branch(self):
        assert relax_exact_match("The answer is B", "B") == 1.0
        assert relax_exact_match("The answer is C", "B") == 0.0
        assert relax_exact_match("B", "B") == 1.0
        assert relax_exact_match("A", "B") == 0.0

    def test_wrapped_letter_gold_keeps_mcq_branch(self):
        assert relax_exact_match("A", "(A)") == 1.0
        assert relax_exact_match("(B)", "(B)") == 1.0
        assert relax_exact_match("B.", "(A)") == 0.0
