"""Cognitive map evaluation module.

This module provides comprehensive evaluation for cognitive mapping tasks including:
- Cognitive map similarity metrics
- Graph-based spatial relationship analysis
- Rotation-invariant comparison
- Validation and error analysis
"""

from .cogmap_evaluator import CogMapEvaluator
from .cogmap_metrics import (
    calculate_cogmap_similarity,
    calculate_extended_cogmap_similarity,
    is_complex_format,
    validate_cogmap_format,
)

__all__ = [
    "calculate_cogmap_similarity",
    "calculate_extended_cogmap_similarity",
    "validate_cogmap_format",
    "is_complex_format",
    "CogMapEvaluator",
]


def evaluate_cogmap_responses(jsonl_path: str, output_path: str = None, include_detailed_metrics: bool = True) -> dict:
    """High-level interface for cognitive map evaluation.

    Args:
        jsonl_path: Path to JSONL file with model responses
        output_path: Optional path to save results
        include_detailed_metrics: Whether to include detailed similarity metrics

    Returns:
        Dictionary with evaluation results and error analysis

    """
    evaluator = CogMapEvaluator(include_detailed_metrics=include_detailed_metrics)
    return evaluator.evaluate(jsonl_path, output_path)


def quick_cogmap_check(jsonl_path: str) -> dict:
    """Quick cognitive map evaluation with basic metrics only.

    Args:
        jsonl_path: Path to JSONL file with model responses

    Returns:
        Dictionary with basic evaluation results

    """
    evaluator = CogMapEvaluator(include_detailed_metrics=False)
    return evaluator.evaluate(jsonl_path)
