from collections import defaultdict
from typing import Dict


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
        "correct": 0,
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


def _initialize_cogmap_results_structure() -> Dict:
    """Initialize results structure with cognitive map specific fields."""
    results = initialize_basic_results_structure()

    # Always include detailed metrics for MindCube task
    include_detailed_metrics = True

    if include_detailed_metrics:
        # Add cognitive map similarity structure
        cogmap_similarity = {
            "total_valid": 0,
            "valid_percent": 0.0,
            "isomorphic_count": 0,  # Backward compatibility
            "rotation_invariant_isomorphic_count": 0,
            "parsable_json_count": 0,
            "valid_format_count": 0,
            "avg_relative_position_accuracy": 0.0,
            "avg_facing_similarity": 0.0,
            "avg_directional_similarity": 0.0,
            "avg_overall_similarity": 0.0,
            "rotation_distribution": defaultdict(int),
        }

        results["cogmap_similarity"] = cogmap_similarity

        # Add to each setting
        for setting in results["settings"]:
            results["settings"][setting]["cogmap_similarity"] = {
                "total_valid": 0,
                "valid_percent": 0.0,
                "isomorphic_count": 0,  # Backward compatibility
                "rotation_invariant_isomorphic_count": 0,
                "parsable_json_count": 0,
                "valid_format_count": 0,
                "avg_relative_position_accuracy": 0.0,
                "avg_facing_similarity": 0.0,
                "avg_directional_similarity": 0.0,
                "avg_overall_similarity": 0.0,
                "rotation_distribution": defaultdict(int),
            }

    return results


def _initialize_similarity_accumulators() -> Dict:
    """Initialize similarity metric accumulators."""
    return {
        "parsable_json_count": 0,
        "valid_format_count": 0,
        "valid_graph_count": 0,
        "isomorphic_count": 0,  # Backward compatibility
        "rotation_invariant_isomorphic_count": 0,
        "total_relative_position_accuracy": 0.0,
        "total_facing_similarity": 0.0,
        "total_directional_similarity": 0.0,
        "total_overall_similarity": 0.0,
    }


def _update_similarity_metrics(
    similarity: Dict,
    results: Dict,
    setting: str,
    total_metrics: Dict,
    include_in_overall: bool,
):
    """Update similarity metrics in results structure."""
    if similarity.get("parsable_json", False):
        results["settings"][setting]["cogmap_similarity"]["parsable_json_count"] += 1
        if include_in_overall:
            total_metrics["parsable_json_count"] += 1

        if similarity.get("valid_format", False):
            results["settings"][setting]["cogmap_similarity"]["valid_format_count"] += 1
            if include_in_overall:
                total_metrics["valid_format_count"] += 1

            if similarity.get("valid_graph", False):
                results["settings"][setting]["cogmap_similarity"]["total_valid"] += 1
                if include_in_overall:
                    total_metrics["valid_graph_count"] += 1

                # Update isomorphic counts (backward compatibility)
                if similarity.get("isomorphic", False):
                    results["settings"][setting]["cogmap_similarity"]["isomorphic_count"] += 1
                    if include_in_overall:
                        total_metrics["isomorphic_count"] += 1

                # Update new rotation-invariant isomorphic count
                if similarity.get("rotation_invariant_isomorphic", False):
                    results["settings"][setting]["cogmap_similarity"]["rotation_invariant_isomorphic_count"] += 1
                    if include_in_overall:
                        total_metrics["rotation_invariant_isomorphic_count"] += 1

                # Update setting-specific similarities (using same field names as old version)
                rel_pos_acc = similarity.get("relative_position_accuracy", 0.0)
                facing_sim = similarity.get("facing_similarity", 0.0)
                dir_sim = similarity.get("directional_similarity", 0.0)
                overall_sim = similarity.get("overall_similarity", 0.0)

                results["settings"][setting]["cogmap_similarity"]["avg_relative_position_accuracy"] += rel_pos_acc
                results["settings"][setting]["cogmap_similarity"]["avg_facing_similarity"] += facing_sim
                results["settings"][setting]["cogmap_similarity"]["avg_directional_similarity"] += dir_sim
                results["settings"][setting]["cogmap_similarity"]["avg_overall_similarity"] += overall_sim

                # Accumulate similarities for filtered metrics
                if include_in_overall:
                    total_metrics["total_relative_position_accuracy"] += rel_pos_acc
                    total_metrics["total_facing_similarity"] += facing_sim
                    total_metrics["total_directional_similarity"] += dir_sim
                    total_metrics["total_overall_similarity"] += overall_sim

                # Track rotation distribution
                if similarity.get("best_rotation") is not None:
                    rotation_name = similarity["best_rotation"].get("name", "none")
                else:
                    rotation_name = "none"

                # Update global rotation distribution only if included in overall
                if include_in_overall:
                    results["cogmap_similarity"]["rotation_distribution"][rotation_name] += 1

                # Update setting-specific rotation distribution
                if "rotation_distribution" not in results["settings"][setting]["cogmap_similarity"]:
                    results["settings"][setting]["cogmap_similarity"]["rotation_distribution"] = defaultdict(int)
                results["settings"][setting]["cogmap_similarity"]["rotation_distribution"][rotation_name] += 1


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
        if stats.get("include_in_overall", True):  # Only include if explicitly marked as True
            filtered_total += stats["total"]
            filtered_correct += stats["gen_cogmap_correct"]

    return filtered_total, filtered_correct


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


def _preserve_necessary_cogmap_fields(results: Dict) -> Dict:
    """Preserve necessary fields in the results dictionary."""
    new_cogmap_results = {
        "parsable_json_count": results["cogmap_similarity"]["parsable_json_count"],
        "parsable_json_accuracy": round(results["cogmap_similarity"]["parsable_json_count"] / results["total"], 4) if results["total"] > 0.0001 else 0.0,
        "valid_count": results["cogmap_similarity"]["total_valid"],
        "valid_accuracy": round(results["cogmap_similarity"]["total_valid"] / results["total"], 4) if results["total"] > 0.0001 else 0.0,
        "isomorphic_count": results["cogmap_similarity"]["isomorphic_count"],
        "isomorphic_accuracy": round(results["cogmap_similarity"]["isomorphic_count"] / results["total"], 4) if results["total"] > 0.0001 else 0.0,
        "avg_overall_similarity": round(results["cogmap_similarity"]["avg_overall_similarity"], 4),
        "avg_facing_similarity": round(results["cogmap_similarity"]["avg_facing_similarity"], 4),
        "avg_directional_similarity": round(results["cogmap_similarity"]["avg_directional_similarity"], 4),
        "rotation_distribution": results["cogmap_similarity"]["rotation_distribution"],
    }
    results["cogmap_similarity"] = new_cogmap_results
    results["gen_cogmap_accuracy"] = round(results["gen_cogmap_accuracy"], 4)
    return results
