"""Main evaluation interface with automatic mode selection.

This module provides a unified interface that automatically selects the appropriate
evaluation mode (basic or cognitive map) based on the task type.
"""

from typing import Dict, Literal, Optional

from .cogmap import CogMapEvaluator
from .core.base_metrics import (
    initialize_basic_results_structure,
    update_accuracy_metrics,
)
from .core.extractors import extract_answer, get_setting_from_id
from .core.io_utils import load_jsonl_data, print_basic_results, save_json_results

TaskType = Literal["basic", "cogmap", "cognitive_map"]


class BasicEvaluator:
    """Basic task evaluator for answer accuracy only.

    This evaluator focuses only on extracting answers and calculating accuracy,
    without any cognitive map processing.
    """

    def evaluate(self, jsonl_path: str, output_path: Optional[str] = None) -> Dict:
        """Run basic evaluation with answer accuracy only.

        Args:
            jsonl_path: Path to JSONL file with model responses
            output_path: Optional path to save results

        Returns:
            Dictionary with basic evaluation results

        """
        # Load data
        data = load_jsonl_data(jsonl_path)

        # Initialize results
        results = initialize_basic_results_structure()
        results["unfiltered_total"] = len(data)

        # Initialize error tracking
        error_cases = {
            "gen_cogmap_error": [],
        }

        # Process each item
        filtered_total = 0
        filtered_correct = 0

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
            if include_in_overall:
                filtered_total += 1

            # Extract answer from response
            answer_text = item.get("cogmap_gen_answer", item.get("answer", ""))
            extracted_answer = extract_answer(answer_text)

            # Track cases where no answer could be extracted
            if not extracted_answer:
                error_cases["gen_cogmap_error"].append(
                    {
                        "id": item_id,
                        "question": item.get("question", ""),
                        "gt_answer": gt_answer,
                        "answer": answer_text,
                    }
                )

            # Compare with ground truth
            is_correct = extracted_answer == gt_answer if extracted_answer else False

            # Update statistics
            if is_correct:
                results["settings"][setting]["gen_cogmap_correct"] += 1
                if include_in_overall:
                    filtered_correct += 1

        # Update final metrics
        results["total"] = filtered_total
        results["gen_cogmap_correct"] = filtered_correct

        # Update accuracy metrics
        results = update_accuracy_metrics(results)

        # Create final result structure
        final_results = {"results": results, "error_cases": error_cases}

        # Print results
        print_basic_results(results)

        # Save results if output path provided
        if output_path:
            save_json_results(final_results, output_path)

        return final_results


def evaluate(
    jsonl_path: str,
    task_type: TaskType = "basic",
    output_path: Optional[str] = None,
    **kwargs,
) -> Dict:
    """Unified evaluation interface with automatic mode selection.

    Args:
        jsonl_path: Path to JSONL file with model responses
        task_type: Type of task - "basic" for answer accuracy only,
                   "cogmap"/"cognitive_map" for full cognitive map evaluation
        output_path: Optional path to save results
        **kwargs: Additional arguments passed to the specific evaluator

    Returns:
        Dictionary with evaluation results

    Examples:
        # Basic evaluation (answer accuracy only)
        results = evaluate("responses.jsonl", "basic")

        # Full cognitive map evaluation
        results = evaluate("responses.jsonl", "cogmap")

        # Quick cognitive map check (no detailed metrics)
        results = evaluate("responses.jsonl", "cogmap", include_detailed_metrics=False)

    """
    if task_type in ["cogmap", "cognitive_map"]:
        # Use cognitive map evaluator
        include_detailed_metrics = kwargs.get("include_detailed_metrics", True)
        evaluator = CogMapEvaluator(include_detailed_metrics=include_detailed_metrics)
        return evaluator.evaluate(jsonl_path, output_path)

    elif task_type == "basic":
        # Use basic evaluator
        evaluator = BasicEvaluator()
        return evaluator.evaluate(jsonl_path, output_path)

    else:
        raise ValueError(f"Unknown task type: {task_type}. Use 'basic', 'cogmap', or 'cognitive_map'")


def auto_evaluate(jsonl_path: str, output_path: Optional[str] = None) -> Dict:
    """Automatically determine evaluation mode based on data content.

    This function examines the data to determine if it contains cognitive maps
    and automatically selects the appropriate evaluation mode.

    Args:
        jsonl_path: Path to JSONL file with model responses
        output_path: Optional path to save results

    Returns:
        Dictionary with evaluation results

    """
    # Load a sample of data to determine content type
    data = load_jsonl_data(jsonl_path)

    if not data:
        raise ValueError("No data found in the input file")

    # Check first few items for cognitive map content
    has_cogmap = False
    for item in data[: min(5, len(data))]:
        if "grounded_cogmap" in item or "cogmap" in item:
            has_cogmap = True
            break

        # Check if the response contains JSON (potential cognitive map)
        response = item.get("cogmap_gen_answer", item.get("answer", ""))
        if response and ("{" in response and "}" in response):
            has_cogmap = True
            break

    # Select appropriate evaluation mode
    if has_cogmap:
        print("🗺️  Detected cognitive map content - using full cognitive map evaluation")
        return evaluate(jsonl_path, "cogmap", output_path)
    else:
        print("📝 No cognitive map content detected - using basic evaluation")
        return evaluate(jsonl_path, "basic", output_path)
