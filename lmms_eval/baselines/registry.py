"""Baseline Registry for Model Comparison.

This registry maps model presets to their baseline results across different benchmarks.
Structure supports two dimensions: model × task.

Registry Structure:
    BASELINE_REGISTRY = {
        "<model_name>": {
            "_meta": {                          # Optional metadata
                "model": "Full Model Name",
                "hf_repo": "default/hf/repo",   # Default HF repo for this model
            },
            "<task_name>": {                    # Task-specific baseline
                "hf_url": "hf://user/repo/file.jsonl",
                "description": "Description",
            },
            ...
        },
    }

Usage:
    --baseline qwen25vl           # Auto-match current task (e.g., videomme)
    --baseline qwen25vl:mmbench   # Explicitly specify task (override)

To add a new baseline:
    1. Add model entry if not exists
    2. Add task entry under the model with hf_url pointing to the JSONL file
"""

from typing import Any, Dict

BASELINE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "qwen25vl": {
        "_meta": {
            "model": "Qwen2.5-VL-7B-Instruct",
            "hf_repo": "mwxely/lmms-eval-test",
        },
        "videomme": {
            "hf_url": "hf://mwxely/lmms-eval-test/20251111_202127_samples_videomme_w_subtitle.jsonl",
            "description": "VideoMME w/ subtitle",
        },
    },
}
