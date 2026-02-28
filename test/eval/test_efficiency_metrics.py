"""Tests for efficiency metrics aggregation."""

import pytest

from lmms_eval.models.model_utils.efficiency_metrics import build_efficiency_summary


@pytest.fixture
def base_results():
    """Fixture providing base results structure for efficiency metric tests."""
    return {
        "config": {
            "model": "openai",
            "model_args": "model_version=gpt-5.2-mini",
        },
        "configs": {
            "toy_task": {
                "score_key": "score",
            }
        },
        "samples": {
            "toy_task": [
                {
                    "token_counts": [{"input_tokens": 100, "output_tokens": 20}],
                    "score": 1,
                },
                {
                    "token_counts": [{"input_tokens": 50, "output_tokens": 30}],
                    "score": 0,
                },
            ]
        },
    }


def test_efficiency_metrics_tokens_per_correct_without_pricing(base_results):
    """Test token counting and tokens-per-correct calculation without pricing."""
    # Arrange
    summary = build_efficiency_summary(base_results)

    # Act & Assert
    assert "overall" in summary
    assert summary["overall"]["total_input_tokens"] == 150.0
    assert summary["overall"]["total_output_tokens"] == 50.0
    assert summary["overall"]["total_correct_score"] == 1.0
    assert summary["overall"]["tokens_per_correct_answer"] == 50.0
    assert "pricing" not in summary


def test_efficiency_metrics_empty_samples_returns_empty_summary():
    """Test that empty samples returns empty summary dict."""
    # Arrange
    results = {"samples": {}}

    # Act
    summary = build_efficiency_summary(results)

    # Assert
    assert summary == {}


def test_efficiency_metrics_score_fallback_from_acc_when_score_key_missing(base_results):
    """Test fallback to 'acc' field when configured score_key is missing."""
    # Arrange
    base_results["configs"]["toy_task"]["score_key"] = "custom_metric"
    base_results["samples"]["toy_task"][0].pop("score")
    base_results["samples"]["toy_task"][0]["acc"] = 1
    base_results["samples"]["toy_task"][1].pop("score")
    base_results["samples"]["toy_task"][1]["acc"] = 0

    # Act
    summary = build_efficiency_summary(base_results)

    # Assert
    assert summary["overall"]["total_correct_score"] == 1.0
    assert summary["overall"]["tokens_per_correct_answer"] == 50.0


def test_efficiency_metrics_ignores_non_dict_token_entries(base_results):
    """Test that non-dict entries in token_counts are silently skipped."""
    # Arrange
    base_results["samples"]["toy_task"][0]["token_counts"] = [
        None,
        {"input_tokens": 100, "output_tokens": 20},
    ]
    base_results["samples"]["toy_task"][1]["token_counts"] = [
        "bad",
        {"input_tokens": 50, "output_tokens": 30},
    ]

    # Act
    summary = build_efficiency_summary(base_results)

    # Assert
    assert summary["overall"]["total_input_tokens"] == 150.0
    assert summary["overall"]["total_output_tokens"] == 50.0


def test_efficiency_metrics_zero_correct_score_returns_none_tokens_per_correct(
    base_results,
):
    """Test that tokens_per_correct_answer is None when total_correct_score is zero."""
    # Arrange
    base_results["samples"]["toy_task"][0]["score"] = 0
    base_results["samples"]["toy_task"][1]["score"] = 0

    # Act
    summary = build_efficiency_summary(base_results)

    # Assert
    assert summary["overall"]["total_correct_score"] == 0.0
    assert summary["overall"]["tokens_per_correct_answer"] is None
