"""Cognitive map evaluator for orchestrating complete evaluation workflows.

This module provides the CogMapEvaluator class that handles:
1. Loading and processing evaluation data
2. Running cognitive map similarity analysis
3. Generating comprehensive results and error reports
"""

from collections import defaultdict
from typing import Dict, Optional

from ..core.base_metrics import (
    apply_filtering_to_results,
    initialize_basic_results_structure,
)
from ..core.extractors import (
    determine_answer_fields,
    extract_answer,
    extract_json_from_text,
    get_setting_from_id,
)
from ..core.io_utils import load_jsonl_data, print_basic_results, save_json_results
from .cogmap_metrics import calculate_cogmap_similarity


class CogMapEvaluator:
    """Comprehensive cognitive map evaluator.

    This class provides a complete evaluation workflow for cognitive mapping tasks,
    including answer accuracy and spatial relationship similarity analysis.
    """

    def __init__(self, include_detailed_metrics: bool = True):
        """Initialize the cognitive map evaluator.

        Args:
            include_detailed_metrics: Whether to include detailed similarity metrics

        """
        self.include_detailed_metrics = include_detailed_metrics

    def _preserve_necessary_cogmap_fields(self, results: Dict) -> Dict:
        """Preserve necessary fields in the results dictionary."""
        new_cogmap_results = {
            "parsable_json_count": results["cogmap_similarity"]["parsable_json_count"],
            "parsable_json_accuracy": round(
                results["cogmap_similarity"]["parsable_json_count"] / results["total"],
                4,
            ),
            "valid_count": results["cogmap_similarity"]["total_valid"],
            "valid_accuracy": round(results["cogmap_similarity"]["total_valid"] / results["total"], 4),
            "isomorphic_count": results["cogmap_similarity"]["isomorphic_count"],
            "isomorphic_accuracy": round(results["cogmap_similarity"]["isomorphic_count"] / results["total"], 4),
            "avg_overall_similarity": round(results["cogmap_similarity"]["avg_overall_similarity"], 4),
            "avg_facing_similarity": round(results["cogmap_similarity"]["avg_facing_similarity"], 4),
            "avg_directional_similarity": round(results["cogmap_similarity"]["avg_directional_similarity"], 4),
            "rotation_distribution": results["cogmap_similarity"]["rotation_distribution"],
        }
        results["cogmap_similarity"] = new_cogmap_results
        results["gen_cogmap_accuracy"] = round(results["gen_cogmap_accuracy"], 4)
        return results

    def evaluate(self, jsonl_path: str, output_path: Optional[str] = None) -> Dict:
        """Run complete cognitive map evaluation.

        Args:
            jsonl_path: Path to JSONL file with model responses
            output_path: Optional path to save results

        Returns:
            Dictionary with evaluation results and error analysis

        """
        # Load data
        data = load_jsonl_data(jsonl_path)

        # Initialize results
        results = self._initialize_cogmap_results_structure()

        # Initialize error tracking
        error_cases = {"gen_cogmap_error": [], "cogmap_extraction_error": []}

        # Process each item
        # Accumulators for cognitive map metrics (only if detailed metrics enabled)
        if self.include_detailed_metrics:
            total_similarity_metrics = self._initialize_similarity_accumulators()

        for item in data:
            gt_answer = item.get("gt_answer")
            if not gt_answer:
                continue

            # Extract setting from item ID
            item_id = item.get("id", "")
            setting = get_setting_from_id(item_id)
            results["settings"][setting]["total"] += 1

            # Check if this setting should be included in overall metrics
            include_in_overall = results["settings"][setting].get("include_in_overall", True)

            # Determine which fields contain the answers
            cogmap_field, plain_field = determine_answer_fields(item)

            # Process answer
            cogmap_answer = item.get(cogmap_field, "")
            extracted_answer = extract_answer(cogmap_answer)

            # Track cases where no answer could be extracted
            if not extracted_answer:
                error_cases["gen_cogmap_error"].append(
                    {
                        "id": item_id,
                        "question": item.get("question", ""),
                        "gt_answer": gt_answer,
                        "answer": cogmap_answer,
                    }
                )

            # Compare with ground truth
            is_correct = extracted_answer == gt_answer if extracted_answer else False

            # Update statistics
            if is_correct:
                results["settings"][setting]["gen_cogmap_correct"] += 1

            # Process cognitive maps if detailed metrics are enabled
            if self.include_detailed_metrics:
                generated_cogmap = self._extract_cognitive_map(cogmap_answer, item, cogmap_field, item_id, error_cases)
                grounded_cogmap = self._extract_grounded_cogmap(item)

                if generated_cogmap and grounded_cogmap:
                    similarity = calculate_cogmap_similarity(generated_cogmap, grounded_cogmap)
                    self._update_similarity_metrics(
                        similarity,
                        results,
                        setting,
                        total_similarity_metrics,
                        include_in_overall,
                    )

        # Apply filtering logic (exclude translation from overall metrics)
        results = apply_filtering_to_results(results)

        # Finalize cognitive map metrics if detailed metrics enabled
        if self.include_detailed_metrics:
            self._finalize_cogmap_metrics(results, total_similarity_metrics)

            results = self._preserve_necessary_cogmap_fields(results)

        # Create final result structure
        final_results = {"results": results, "error_cases": error_cases}

        # Print results
        self._print_results(results)

        # Save results if output path provided
        if output_path:
            save_json_results(final_results, output_path)

        return final_results

    def _initialize_cogmap_results_structure(self) -> Dict:
        """Initialize results structure with cognitive map specific fields."""
        results = initialize_basic_results_structure()

        if self.include_detailed_metrics:
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

    def _initialize_similarity_accumulators(self) -> Dict:
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

    def _extract_cognitive_map(
        self,
        cogmap_answer: str,
        item: Dict,
        cogmap_field: str,
        item_id: str,
        error_cases: Dict,
    ) -> Optional[Dict]:
        """Extract cognitive map from response with error handling."""
        try:
            # First try direct extraction from the answer text
            generated_cogmap = extract_json_from_text(cogmap_answer)

            # If extraction failed, check for separately stored cognitive map
            if not generated_cogmap:
                if "cognitive_map" in item:
                    cognitive_map = item.get("cognitive_map")
                    if isinstance(cognitive_map, str):
                        generated_cogmap = extract_json_from_text(cognitive_map)
                    else:
                        generated_cogmap = cognitive_map
                elif cogmap_field + "_cognitive_map" in item:
                    cognitive_map = item.get(cogmap_field + "_cognitive_map")
                    if isinstance(cognitive_map, str):
                        generated_cogmap = extract_json_from_text(cognitive_map)
                    else:
                        generated_cogmap = cognitive_map
                elif "grounded_cogmap" in item and item["grounded_cogmap"] is not None:
                    if cogmap_field.startswith("gen") or cogmap_field.startswith("cogmap_gen"):
                        cognitive_map = item.get("grounded_cogmap")
                        if isinstance(cognitive_map, str):
                            generated_cogmap = extract_json_from_text(cognitive_map)
                        else:
                            generated_cogmap = cognitive_map

            return generated_cogmap

        except Exception as e:
            error_cases["cogmap_extraction_error"].append(
                {
                    "id": item_id,
                    "type": "generated",
                    "response": cogmap_answer[:500],  # Truncate for readability
                    "error": str(e),
                }
            )
            return None

    def _extract_grounded_cogmap(self, item: Dict) -> Optional[Dict]:
        """Extract grounded cognitive map from item."""
        grounded_cogmap = item.get("grounded_cogmap")
        if isinstance(grounded_cogmap, str):
            grounded_cogmap = extract_json_from_text(grounded_cogmap)
        return grounded_cogmap

    def _update_similarity_metrics(
        self,
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

    def _finalize_cogmap_metrics(self, results: Dict, total_metrics: Dict):
        """Finalize cognitive map metrics by calculating averages and percentages."""
        filtered_total = results["total"]  # This is now the filtered total
        filtered_valid_cogmap_count = total_metrics["valid_graph_count"]

        if filtered_total > 0:
            # Overall metrics
            results["cogmap_similarity"]["parsable_json_count"] = total_metrics["parsable_json_count"]
            results["cogmap_similarity"]["valid_format_count"] = total_metrics["valid_format_count"]
            results["cogmap_similarity"]["total_valid"] = filtered_valid_cogmap_count
            results["cogmap_similarity"]["valid_percent"] = (filtered_valid_cogmap_count / filtered_total) * 100
            results["cogmap_similarity"]["isomorphic_count"] = total_metrics["isomorphic_count"]  # Backward compatibility
            results["cogmap_similarity"]["rotation_invariant_isomorphic_count"] = total_metrics["rotation_invariant_isomorphic_count"]

            # Calculate averages for overall metrics (using same logic as old version)
            if filtered_valid_cogmap_count > 0:
                results["cogmap_similarity"]["avg_relative_position_accuracy"] = total_metrics["total_relative_position_accuracy"] / filtered_valid_cogmap_count
                results["cogmap_similarity"]["avg_facing_similarity"] = total_metrics["total_facing_similarity"] / filtered_valid_cogmap_count
                results["cogmap_similarity"]["avg_directional_similarity"] = total_metrics["total_directional_similarity"] / filtered_valid_cogmap_count
                results["cogmap_similarity"]["avg_overall_similarity"] = total_metrics["total_overall_similarity"] / filtered_valid_cogmap_count

        # Setting-specific metrics (using same logic as old version)
        for setting, stats in results["settings"].items():
            setting_total = stats["total"]
            setting_valid = stats["cogmap_similarity"]["total_valid"]

            if setting_total > 0:
                stats["cogmap_similarity"]["valid_percent"] = (setting_valid / setting_total) * 100

            if setting_valid > 0:
                stats["cogmap_similarity"]["avg_relative_position_accuracy"] /= setting_valid
                stats["cogmap_similarity"]["avg_facing_similarity"] /= setting_valid
                stats["cogmap_similarity"]["avg_directional_similarity"] /= setting_valid
                stats["cogmap_similarity"]["avg_overall_similarity"] /= setting_valid

    def _print_results(self, results: Dict):
        """Print results with cognitive map specific information."""
        print_basic_results(results)

        if self.include_detailed_metrics and "cogmap_similarity" in results:
            self._print_cogmap_metrics(results)

    def _print_cogmap_metrics(self, results: Dict):
        """Print cognitive map specific metrics."""
        cogmap_sim = results["cogmap_similarity"]
        total = results["total"]
        unfiltered_total = results.get("unfiltered_total", total)

        print("\n=== COGNITIVE MAP METRICS ===")
        print(f"Total examples: {unfiltered_total} (Evaluated: {total}, excluding translation)")

        # Validation metrics
        parsable_count = cogmap_sim.get("parsable_json_count", 0)
        valid_format_count = cogmap_sim.get("valid_format_count", 0)
        valid_graph_count = cogmap_sim.get("total_valid", 0)

        print(f"Parsable JSON: {parsable_count}/{total} ({100 * parsable_count / total:.1f}%)")
        print(f"Valid format: {valid_format_count}/{total} ({100 * valid_format_count / total:.1f}%)")
        print(f"Valid graphs: {valid_graph_count}/{total} ({cogmap_sim.get('valid_percent', 0):.1f}%)")

        # Similarity metrics (only if we have valid graphs)
        if valid_graph_count > 0:
            # Use isomorphic_count for consistency with old version
            iso_count = cogmap_sim.get("isomorphic_count", 0)
            print(f"Isomorphic graphs: {iso_count}/{total} ({100 * iso_count / total:.2f}%)")

            print("\nAverage similarities:")
            # Convert from decimal to percentage for display (matching old version)
            print(f"  Directional: {cogmap_sim.get('avg_directional_similarity', 0) * 100:.2f}%")
            print(f"  Facing: {cogmap_sim.get('avg_facing_similarity', 0) * 100:.2f}%")
            print(f"  Overall: {cogmap_sim.get('avg_overall_similarity', 0) * 100:.2f}%")

            # Print rotation distribution
            rotation_dist = cogmap_sim.get("rotation_distribution", {})
            if rotation_dist:
                print(f"Rotation distribution: {dict(rotation_dist)}")
                # Detailed rotation distribution (optional)
                # print(f"\nRotation distribution:")
                # for rotation, count in sorted(rotation_dist.items()):
                #     percentage = (count / valid_graph_count) * 100
                #     print(f"  {rotation}: {count} ({percentage:.1f}%)")
