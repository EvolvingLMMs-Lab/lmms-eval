"""Base metrics calculation for MindCube evaluation.

This module provides core evaluation functions for calculating accuracy and
organizing results by task settings.
"""

from typing import Dict


def calculate_accuracy(results: Dict) -> float:
    """Calculate accuracy percentage from results.

    Args:
        results: Results dictionary

    Returns:
        Accuracy as percentage (0-100)

    """
    return results.get("gen_cogmap_accuracy", 0.0) * 100


def initialize_basic_results_structure() -> Dict:
    """Initialize the basic results data structure.

    Returns:
        Empty results structure with all required fields

    """
    # Basic settings to track - maintain consistent order with original
    settings = ["around", "rotation", "translation", "among", "other"]

    # Define which settings should be included in overall metrics
    # Translation is excluded from overall metrics but still tracked
    settings_to_include = {
        "around": True,
        "rotation": True,
        "translation": False,  # Exclude translation from overall metrics
        "among": True,
        "other": True,
    }

    # Initialize results
    results = {
        "total": 0,
        "unfiltered_total": 0,  # Original total (all settings)
        "gen_cogmap_correct": 0,
        "gen_cogmap_accuracy": 0.0,
        "settings": {
            setting: {
                "total": 0,
                "gen_cogmap_correct": 0,
                "gen_cogmap_accuracy": 0.0,
                "include_in_overall": settings_to_include.get(setting, True),  # Flag for filtering
            }
            for setting in settings
        },
    }

    return results


def update_accuracy_metrics(results: Dict) -> Dict:
    """Update accuracy metrics for all settings.

    Args:
        results: Results dictionary to update

    Returns:
        Updated results dictionary

    """
    # Calculate setting-specific accuracy
    for setting, stats in results["settings"].items():
        if stats["total"] > 0:
            stats["gen_cogmap_accuracy"] = stats["gen_cogmap_correct"] / stats["total"]
        else:
            stats["gen_cogmap_accuracy"] = 0.0

    # Calculate overall accuracy using filtered total
    if results["total"] > 0:
        results["gen_cogmap_accuracy"] = results["gen_cogmap_correct"] / results["total"]
    else:
        results["gen_cogmap_accuracy"] = 0.0

    return results


def get_filtered_totals(results: Dict) -> tuple:
    """Calculate filtered totals excluding settings that shouldn't be included in overall metrics.

    Args:
        results: Results dictionary

    Returns:
        Tuple of (filtered_total, filtered_correct)

    """
    filtered_total = 0
    filtered_correct = 0

    for setting, stats in results["settings"].items():
        if stats.get("include_in_overall", True):
            filtered_total += stats["total"]
            filtered_correct += stats["gen_cogmap_correct"]

    return filtered_total, filtered_correct


def get_unfiltered_totals(results: Dict) -> tuple:
    """Calculate unfiltered totals including all settings.

    Args:
        results: Results dictionary

    Returns:
        Tuple of (unfiltered_total, unfiltered_correct)

    """
    unfiltered_total = 0
    unfiltered_correct = 0

    for setting, stats in results["settings"].items():
        unfiltered_total += stats["total"]
        unfiltered_correct += stats["gen_cogmap_correct"]

    return unfiltered_total, unfiltered_correct


def apply_filtering_to_results(results: Dict) -> Dict:
    """Apply filtering logic to exclude certain settings from overall metrics.
    This maintains compatibility with the original evaluation logic.

    Args:
        results: Results dictionary to filter

    Returns:
        Filtered results dictionary

    """
    # Calculate unfiltered totals (all settings)
    unfiltered_total, unfiltered_correct = get_unfiltered_totals(results)
    results["unfiltered_total"] = unfiltered_total

    # Calculate filtered totals (excluding translation)
    filtered_total, filtered_correct = get_filtered_totals(results)

    # Update overall metrics with filtered values
    results["total"] = filtered_total
    results["gen_cogmap_correct"] = filtered_correct

    # Recalculate overall accuracy
    results = update_accuracy_metrics(results)

    return results
