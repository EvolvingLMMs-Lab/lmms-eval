"""Baseline comparison module for paired t-test analysis.

This module provides functionality for comparing model performance against
baseline results using paired t-test statistical analysis.

Usage:
    --baseline qwen25vl           # Auto-match current task from registry
    --baseline qwen25vl:mmbench   # Explicitly specify task
    --baseline /path/to/file.jsonl    # Local file
    --baseline hf://user/repo/file    # Direct HF URL
"""

from lmms_eval.baselines.loader import load_baseline
from lmms_eval.baselines.registry import BASELINE_REGISTRY

__all__ = ["BASELINE_REGISTRY", "load_baseline"]
