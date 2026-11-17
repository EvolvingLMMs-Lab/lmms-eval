"""Tests for the ``median`` aggregation and ``power_analysis`` input validation."""

import pytest

from lmms_eval.api.metrics import median, power_analysis


class TestMedianAggregation:
    def test_unsorted_odd_length(self):
        assert median([5, 1, 9]) == 5

    def test_unsorted_even_length_averages_middle_pair(self):
        assert median([3, 1, 2, 10]) == 2.5

    def test_single_element(self):
        assert median([7]) == 7

    def test_does_not_mutate_input(self):
        scores = [3, 1, 2]
        median(scores)
        assert scores == [3, 1, 2]


class TestPowerAnalysisValidation:
    def test_valid_input_still_works(self):
        result = power_analysis(effect_size=0.03)
        assert result["min_n"] > 0

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"effect_size": 0.0}, "effect_size"),
            ({"effect_size": float("nan")}, "effect_size"),
            ({"effect_size": 0.03, "alpha": 0.0}, "alpha"),
            ({"effect_size": 0.03, "alpha": 1.0}, "alpha"),
            ({"effect_size": 0.03, "power": 1.0}, "power"),
            ({"effect_size": 0.03, "correlation": 1.5}, "correlation"),
            ({"effect_size": 0.03, "std_a": -0.1}, "non-negative"),
        ],
    )
    def test_invalid_parameters_raise_value_error(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            power_analysis(**kwargs)

    def test_perfect_correlation_with_equal_std_raises_value_error(self):
        with pytest.raises(ValueError, match="zero variance"):
            power_analysis(effect_size=0.03, std_a=0.5, correlation=1.0)
