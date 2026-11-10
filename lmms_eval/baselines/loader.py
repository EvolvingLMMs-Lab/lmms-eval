"""Baseline loading utilities for Model Comparison.

Supports loading baseline results from:
1. Registry presets (model name or model:task)
2. Local JSONL files
3. HuggingFace Hub datasets
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from lmms_eval.baselines.registry import BASELINE_REGISTRY

try:
    from loguru import logger as eval_logger
except ImportError:
    import logging

    eval_logger = logging.getLogger(__name__)


def load_baseline(baseline_arg: str, task_name: str) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Load baseline results from local path, HF URL, or preset name.

    Args:
        baseline_arg: One of:
            - Model preset: "qwen25vl" (auto-match task from BASELINE_REGISTRY)
            - Model:task preset: "qwen25vl:mmbench" (explicit task)
            - Local path: "/path/to/results.jsonl"
            - HF URL: "hf://user/repo/file.jsonl"
        task_name: Current task name for auto-matching presets

    Returns:
        Tuple of (sample rows with a doc_id, optional aggregated results dict)
    """
    # Check for explicit model:task format
    if ":" in baseline_arg and not baseline_arg.startswith("hf://"):
        model_name, explicit_task = baseline_arg.split(":", 1)
        if model_name in BASELINE_REGISTRY:
            return _load_from_registry(model_name, explicit_task, baseline_arg)

    # Check for model preset (auto-match task)
    if baseline_arg in BASELINE_REGISTRY:
        return _load_from_registry(baseline_arg, task_name, baseline_arg)

    # Direct HF URL
    if baseline_arg.startswith("hf://") or "huggingface.co" in baseline_arg:
        return _load_baseline_from_hf(baseline_arg, task_name)

    # Local path
    if os.path.exists(baseline_arg):
        return _load_baseline_from_local(baseline_arg, task_name)

    raise ValueError(f"Cannot load baseline '{baseline_arg}'. Available presets: {list(BASELINE_REGISTRY.keys())}")


def _load_from_registry(model_name: str, task_name: str, baseline_arg: str) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Load baseline from registry by model and task name."""
    model_entry = BASELINE_REGISTRY[model_name]

    # Check if task exists for this model
    if task_name not in model_entry:
        available_tasks = [k for k in model_entry.keys() if not k.startswith("_")]
        raise ValueError(f"No baseline for model '{model_name}' on task '{task_name}'. Available tasks: {available_tasks}")

    task_entry = model_entry[task_name]
    eval_logger.info(f"[Baseline] Using preset '{model_name}' for task '{task_name}'")

    if "hf_url" in task_entry:
        return _load_baseline_from_hf(task_entry["hf_url"], task_name)
    elif "path" in task_entry:
        return _load_baseline_from_local(task_entry["path"], task_name)
    else:
        raise ValueError(f"Preset '{baseline_arg}' has no 'hf_url' or 'path'")


def _load_baseline_from_local(path: str, task_name: str) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Load baseline from local JSONL file."""
    eval_logger.info(f"[Baseline] Loading from: {path}")
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            sample = json.loads(line)
            if sample.get("doc_id") is not None:
                samples.append(sample)
    eval_logger.info(f"[Baseline] Loaded {len(samples)} samples")

    # Try to load aggregated results
    agg_results = None
    dir_path = os.path.dirname(path)
    base_name = os.path.basename(path)
    parts = base_name.split("_samples_")
    if len(parts) == 2:
        results_path = os.path.join(dir_path, parts[0] + "_results.json")
        if os.path.exists(results_path):
            with open(results_path, "r") as f:
                agg_results = json.load(f)
    return samples, agg_results


def _load_baseline_from_hf(hf_path: str, task_name: str) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Load baseline from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download, list_repo_files

    # Parse HF path: hf://user/repo/file.jsonl or hf://user/repo
    if hf_path.startswith("hf://"):
        path_parts = hf_path[5:].split("/")
        if len(path_parts) >= 3:
            # hf://user/repo/file.jsonl -> download specific file
            repo_id = "/".join(path_parts[:2])
            filename = "/".join(path_parts[2:])
            eval_logger.info(f"[Baseline] Loading from HF: {repo_id}/{filename}")
            local_path = hf_hub_download(repo_id, filename, repo_type="dataset")
            return _load_baseline_from_local(local_path, task_name)
        else:
            repo_id = "/".join(path_parts)
    else:
        repo_id = hf_path.split("huggingface.co/datasets/")[-1].rstrip("/")

    eval_logger.info(f"[Baseline] Loading from HF: {repo_id}")
    files = list_repo_files(repo_id, repo_type="dataset")
    jsonl_files = [f for f in files if f.endswith(".jsonl")]
    if not jsonl_files:
        raise ValueError(f"No JSONL files in HF repo: {repo_id}")
    local_path = hf_hub_download(repo_id, jsonl_files[0], repo_type="dataset")
    return _load_baseline_from_local(local_path, task_name)
