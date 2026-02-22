"""I/O utilities for evaluation data loading and result saving.

This module provides functions to:
1. Load data from JSONL files
2. Save evaluation results
3. Print results in a readable format
"""

import glob
import json
import os
from datetime import datetime
from typing import Dict, List


def load_jsonl_data(jsonl_path: str) -> List[Dict]:
    """Load data from a JSONL file.

    Args:
        jsonl_path: Path to the JSONL file

    Returns:
        List of loaded JSON objects

    """
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def save_json_results(results: Dict, output_path: str) -> None:
    """Save evaluation results as JSON.

    Args:
        results: Evaluation results dictionary
        output_path: Path to save the results to

    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_path}")


def find_evaluation_files(eval_dir: str, pattern: str = "*.jsonl") -> List[str]:
    """Find all evaluation files matching a pattern in a directory.

    Args:
        eval_dir: Directory to search in
        pattern: Glob pattern to match

    Returns:
        List of matching file paths

    """
    return glob.glob(os.path.join(eval_dir, pattern))


def create_output_paths(base_name: str, output_dir: str) -> Dict[str, str]:
    """Create standardized output paths for results.

    Args:
        base_name: Base name for the result files
        output_dir: Directory to store the results

    Returns:
        Dictionary of output paths

    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create paths
    paths = {
        "json": os.path.join(output_dir, f"{base_name}_results_{timestamp}.json"),
        "summary": os.path.join(output_dir, f"{base_name}_summary_{timestamp}.txt"),
    }

    return paths


def print_basic_results(results: Dict) -> None:
    """Print basic evaluation results in a readable format.

    Args:
        results: Evaluation results dictionary

    """
    total = results.get("total", 0)
    unfiltered_total = results.get("unfiltered_total", total)
    correct = results.get("gen_cogmap_correct", 0)
    accuracy = results.get("gen_cogmap_accuracy", 0.0)

    print("\n=== EVALUATION RESULTS ===")
    print(f"Total examples: {unfiltered_total} (Evaluated: {total}, excluding translation)")
    print(f"Answer accuracy: {accuracy * 100:.2f}% ({correct}/{total})")

    # Print results by setting
    print("\n=== RESULTS BY SETTING ===")
    for setting, stats in results.get("settings", {}).items():
        setting_total = stats.get("total", 0)
        setting_correct = stats.get("gen_cogmap_correct", 0)
        setting_accuracy = stats.get("gen_cogmap_accuracy", 0.0)
        include_in_overall = stats.get("include_in_overall", True)

        status = "" if include_in_overall else " (excluded from overall)"
        print(f"{setting.capitalize()}: {setting_accuracy * 100:.2f}% ({setting_correct}/{setting_total}){status}")


def print_summary_line(results: Dict, model_name: str = "") -> None:
    """Print a single summary line for quick comparison.

    Args:
        results: Evaluation results dictionary
        model_name: Name of the model being evaluated

    """
    total = results.get("total", 0)
    accuracy = results.get("gen_cogmap_accuracy", 0.0)

    if model_name:
        print(f"{model_name}: {accuracy * 100:.2f}% ({total} examples)")
    else:
        print(f"Accuracy: {accuracy * 100:.2f}% ({total} examples)")
