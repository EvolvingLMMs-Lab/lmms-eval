"""Baseline comparison module for paired t-test analysis.

This module provides functionality for comparing model performance against
baseline results using paired t-test statistical analysis.

Usage:
    --baseline qwen25vl           # Auto-match current task from registry
    --baseline qwen25vl:mmbench   # Explicitly specify task
    --baseline /path/to/file.jsonl    # Local file
    --baseline hf://user/repo/file    # Direct HF URL
"""

import os

from lmms_eval.baselines.loader import load_baseline
from lmms_eval.baselines.registry import BASELINE_REGISTRY


def get_baseline_display_name(baseline_arg: str) -> str:
    """Extract a short display name from baseline argument.

    Args:
        baseline_arg: The baseline argument string, can be:
            - Model preset: "qwen25vl"
            - Model:task format: "qwen25vl:mmbench"
            - HF URL: "hf://user/repo/file.jsonl"
            - Local path: "/path/to/results.jsonl"

    Returns:
        A short, human-readable display name for the baseline.
    """
    # Handle model:task format (e.g., qwen25vl:mmbench)
    if ":" in baseline_arg and not baseline_arg.startswith("hf://"):
        model_name, task = baseline_arg.split(":", 1)
        if model_name in BASELINE_REGISTRY:
            return model_name  # Just show model name
    # Handle model preset (e.g., qwen25vl)
    if baseline_arg in BASELINE_REGISTRY:
        return baseline_arg
    # Handle HF URL
    if baseline_arg.startswith("hf://"):
        # hf://user/repo/file.jsonl -> user/repo
        parts = baseline_arg[5:].split("/")
        return "/".join(parts[:2]) if len(parts) >= 2 else baseline_arg
    # Handle local path
    if "/" in baseline_arg or "\\" in baseline_arg:
        filename = os.path.basename(baseline_arg)
        return os.path.splitext(filename)[0][:30]  # Truncate to 30 chars
    return baseline_arg


__all__ = ["BASELINE_REGISTRY", "get_baseline_display_name", "load_baseline"]
