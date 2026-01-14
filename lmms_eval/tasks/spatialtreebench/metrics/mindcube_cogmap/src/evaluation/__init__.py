"""MindCube Evaluation Framework

A reorganized evaluation system with two main modes:
- Basic evaluation: Answer accuracy only
- Cognitive map evaluation: Full spatial reasoning analysis

Usage Examples:
    # Basic evaluation (answer accuracy only)
    from src.evaluation import evaluate
    results = evaluate("responses.jsonl", "basic")

    # Full cognitive map evaluation
    results = evaluate("responses.jsonl", "cogmap")

    # Auto-detect content type
    from src.evaluation import auto_evaluate
    results = auto_evaluate("responses.jsonl")

    # Quick cognitive map check (no detailed metrics)
    results = evaluate("responses.jsonl", "cogmap", include_detailed_metrics=False)
"""

from .cogmap import CogMapEvaluator, evaluate_cogmap_responses, quick_cogmap_check

# Core utilities (for advanced users)
from .core import (
    calculate_accuracy,
    extract_answer,
    extract_json_from_text,
    load_jsonl_data,
    print_basic_results,
    save_json_results,
)
from .evaluator import BasicEvaluator, auto_evaluate, evaluate

__all__ = [
    # Main interfaces (recommended for most users)
    "evaluate",
    "auto_evaluate",
    # Specific evaluators
    "BasicEvaluator",
    "CogMapEvaluator",
    "evaluate_cogmap_responses",
    "quick_cogmap_check",
    # Core utilities (for advanced users)
    "extract_answer",
    "extract_json_from_text",
    "calculate_accuracy",
    "load_jsonl_data",
    "save_json_results",
    "print_basic_results",
]


def quick_start_guide():
    """Print a quick start guide for using the evaluation framework."""
    print(
        """
🚀 MindCube Evaluation Framework - Quick Start

=== BASIC USAGE ===

1. Basic evaluation (answer accuracy only):
   ```python
   from src.evaluation import evaluate
   results = evaluate("responses.jsonl", "basic")
   ```

2. Full cognitive map evaluation:
   ```python
   results = evaluate("responses.jsonl", "cogmap")
   ```

3. Auto-detect content type:
   ```python
   from src.evaluation import auto_evaluate
   results = auto_evaluate("responses.jsonl")
   ```

=== ADVANCED USAGE ===

4. Quick cognitive map check (fast, basic metrics only):
   ```python
   from src.evaluation import quick_cogmap_check
   results = quick_cogmap_check("responses.jsonl")
   ```

5. Custom cognitive map evaluation:
   ```python
   from src.evaluation import CogMapEvaluator
   evaluator = CogMapEvaluator(include_detailed_metrics=False)
   results = evaluator.evaluate("responses.jsonl")
   ```

6. Batch evaluation:
   ```python
   import glob
   for file in glob.glob("results/*.jsonl"):
       results = evaluate(file, "basic", f"{file}.results.json")
   ```

=== OUTPUT STRUCTURE ===

Results dictionary contains:
- results['results']['gen_cogmap_accuracy']: Overall accuracy
- results['results']['settings']: Per-setting breakdown
- results['error_cases']: Failed extractions for debugging

For cognitive map evaluation, additional metrics include:
- results['results']['cogmap_similarity']: Spatial relationship metrics
- Rotation-invariant isomorphism analysis
- Position and facing similarity scores
"""
    )


# For backwards compatibility with existing scripts
def batch_evaluate(eval_dir: str, output_dir: str = None):
    """Batch evaluate all JSONL files in a directory.

    Args:
        eval_dir: Directory containing JSONL files
        output_dir: Directory to save results (optional)

    """
    import glob
    import os

    pattern = os.path.join(eval_dir, "*.jsonl")
    files = glob.glob(pattern)

    if not files:
        print(f"No JSONL files found in {eval_dir}")
        return

    print(f"Found {len(files)} files to evaluate")

    for file_path in files:
        print(f"\nEvaluating {os.path.basename(file_path)}...")

        # Determine output path
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_results.json")
        else:
            output_path = file_path.replace(".jsonl", "_results.json")

        try:
            # Use auto-detection for batch processing
            results = auto_evaluate(file_path, output_path)
            accuracy = results["results"]["gen_cogmap_accuracy"] * 100
            print(f"✅ {os.path.basename(file_path)}: {accuracy:.1f}% accuracy")

        except Exception as e:
            print(f"❌ Error evaluating {os.path.basename(file_path)}: {e}")

    print(f"\n📁 Results saved to {output_dir or eval_dir}")
