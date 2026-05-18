#!/usr/bin/env python3
"""
Push lmms-eval results JSON files to W&B runs.

The script mirrors lmms-eval wandb argument handling:
  --wandb_args "project=...,entity=...,job_type=...,group=...,name=...,id=...,resume=..."
"""

import argparse
import copy
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

try:
    import wandb
except ModuleNotFoundError:
    wandb = None


EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = str(EXAMPLE_ROOT / "results")
DEFAULT_WANDB_PROJECT = "lmms-eval"
DEFAULT_WANDB_ENTITY = "rkreft-eth-z-rich"
DEFAULT_WANDB_JOB_TYPE = "eval"
DEFAULT_WANDB_RESUME = "must"


def _handle_arg_string(arg: str):
    lower = arg.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if arg.isnumeric():
        return int(arg)
    try:
        return float(arg)
    except ValueError:
        return arg


def _simple_parse_args_string(args_string: str) -> Dict[str, object]:
    if not args_string:
        return {}
    arg_list = [item for item in args_string.split(",") if item]
    args_dict: Dict[str, object] = {}
    for arg in arg_list:
        if "=" not in arg:
            continue
        k, v = arg.split("=", 1)
        args_dict[k] = _handle_arg_string(v)
    return args_dict


def _remove_none_pattern(metric_name: str):
    if metric_name.endswith(",none"):
        return metric_name[: -len(",none")], True
    return metric_name, False


def _model_label_from_dir(model_dir_name: str) -> str:
    if model_dir_name.endswith("__HF"):
        return model_dir_name[: -len("__HF")]
    return model_dir_name


def _deterministic_run_id(seed: str) -> str:
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
    return f"model_{digest}"


def _result_sort_key(path: Path):
    stem = path.stem
    if stem.endswith("_results"):
        stem = stem[: -len("_results")]
    return (stem, path.name)


def _group_results_files_by_model(results_root: Path) -> Dict[Path, List[Path]]:
    grouped: Dict[Path, List[Path]] = defaultdict(list)
    for path in sorted(results_root.glob("*/*_results.json")):
        if path.is_file():
            grouped[path.parent].append(path)
    for parent in list(grouped.keys()):
        grouped[parent] = sorted(grouped[parent], key=_result_sort_key)
    return grouped


def _merge_wandb_args(
    existing_wandb_args: str,
    run_name: str,
    run_id: str,
    project: str,
    entity: str,
    job_type: str,
    resume: str,
    group: str,
) -> str:
    parsed = _simple_parse_args_string(existing_wandb_args) if existing_wandb_args else {}

    # Enforce same defaults/pattern as lmms-eval launcher with explicit run identity.
    parsed["project"] = parsed.get("project") or project
    parsed["entity"] = parsed.get("entity") or entity
    parsed["job_type"] = parsed.get("job_type") or job_type
    if group:
        parsed["group"] = parsed.get("group") or group
    parsed["name"] = run_name
    parsed["id"] = run_id
    # Force resume mode from CLI/script setting.
    parsed["resume"] = resume

    ordered_keys = ["project", "entity", "job_type", "group", "name", "id", "resume"]
    parts: List[str] = []
    for key in ordered_keys:
        value = parsed.get(key)
        if value is None or value == "":
            continue
        parts.append(f"{key}={value}")

    for key in sorted(k for k in parsed.keys() if k not in ordered_keys):
        value = parsed.get(key)
        if value is None or value == "":
            continue
        parts.append(f"{key}={value}")

    return ",".join(parts)


def _sanitize_and_flatten_results(payload: Dict[str, object]):
    results = copy.deepcopy(payload.get("results", {}))
    task_names = list(results.keys())
    wandb_summary: Dict[str, object] = {}

    for task_name in task_names:
        task_result = results.get(task_name, {})
        for metric_name, metric_value in list(task_result.items()):
            cleaned_name, removed = _remove_none_pattern(metric_name)
            if removed:
                task_result[cleaned_name] = metric_value
                task_result.pop(metric_name, None)

    for task_name in task_names:
        task_result = results.get(task_name, {})
        for metric_name, metric_value in list(task_result.items()):
            if isinstance(metric_value, str):
                wandb_summary[f"{task_name}/{metric_name}"] = metric_value
                task_result.pop(metric_name, None)

    flattened: Dict[str, object] = {}
    for task_name in task_names:
        task_result = results.get(task_name, {})
        for metric_name, metric_value in task_result.items():
            flattened[f"{task_name}/{metric_name}"] = metric_value

    return wandb_summary, flattened


def _build_eval_table(payload: Dict[str, object], key: str, first_col: str):
    columns = [first_col, "Version", "Filter", "num_fewshot", "Metric", "Value", "Stderr"]
    table = wandb.Table(columns=columns)
    results = copy.deepcopy(payload)
    group_names = list(results.get("groups", {}).keys())

    for task_or_group, metrics in results.get(key, {}).items():
        if task_or_group in group_names and key != "groups":
            continue
        version = results.get("versions", {}).get(task_or_group)
        nshot = results.get("n-shot", {}).get(task_or_group)

        # Keep table column types stable across rows for W&B Table schema checks.
        version = None if version in (None, "", "N/A") else str(version)
        nshot = None if nshot in (None, "", "N/A") else str(nshot)

        for metric_filter, value in metrics.items():
            metric_name, _, filter_name = metric_filter.partition(",")
            if metric_name.endswith("_stderr") or metric_name == "alias":
                continue

            stderr_key = f"{metric_name}_stderr,{filter_name}"
            stderr = metrics.get(stderr_key, "")
            if stderr != "" and stderr != "N/A":
                try:
                    stderr = f"{float(stderr):.4f}"
                except Exception:
                    pass
            table.add_data(task_or_group, version, filter_name, nshot, metric_name, str(value), str(stderr))

    return table


def _log_results_like_lmms_eval(run, payload: Dict[str, object]):
    task_configs = payload.get("configs", {})
    cli_configs = payload.get("config", {})
    run.config.update({"task_configs": task_configs, "cli_configs": cli_configs}, allow_val_change=True)

    wandb_summary, flattened_results = _sanitize_and_flatten_results(payload)
    run.summary.update(wandb_summary)
    if flattened_results:
        run.log(flattened_results)

    eval_table = _build_eval_table(payload, key="results", first_col="Tasks")
    run.log({"evaluation/eval_results": eval_table})

    if "groups" in payload:
        group_table = _build_eval_table(payload, key="groups", first_col="Groups")
        run.log({"evaluation/group_eval_results": group_table})

    artifact = wandb.Artifact("results", type="eval_results")
    dumped = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    with artifact.new_file("results.json", mode="w", encoding="utf-8") as f:
        f.write(dumped)
    run.log_artifact(artifact)


def _aggregate_latest_payload_for_model(result_files: List[Path], latest_only: bool = False) -> Dict[str, object]:
    if not result_files:
        raise ValueError("result_files cannot be empty")

    sorted_result_files = sorted(result_files, key=_result_sort_key)
    if latest_only:
        with sorted_result_files[-1].open("r", encoding="utf-8") as f:
            return json.load(f)

    latest_payload = None
    merged_results: Dict[str, object] = {}
    merged_versions: Dict[str, object] = {}
    merged_nshot: Dict[str, object] = {}
    merged_groups: Dict[str, object] = {}

    for result_path in sorted_result_files:
        with result_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        latest_payload = payload

        results = payload.get("results", {}) or {}
        versions = payload.get("versions", {}) or {}
        nshots = payload.get("n-shot", {}) or {}
        for task_name, task_metrics in results.items():
            merged_results[task_name] = copy.deepcopy(task_metrics)
            if task_name in versions:
                merged_versions[task_name] = versions[task_name]
            if task_name in nshots:
                merged_nshot[task_name] = nshots[task_name]

        groups = payload.get("groups", {}) or {}
        for group_name, group_metrics in groups.items():
            merged_groups[group_name] = copy.deepcopy(group_metrics)

    merged = copy.deepcopy(latest_payload)
    merged["results"] = merged_results
    merged["versions"] = merged_versions
    merged["n-shot"] = merged_nshot
    if merged_groups:
        merged["groups"] = merged_groups
    elif "groups" in merged:
        merged["groups"] = {}
    return merged


def _push_one_model_aggregate(
    model_dir: Path,
    result_files: List[Path],
    project: str,
    entity: str,
    job_type: str,
    resume: str,
    group_override: str,
    latest_only: bool,
    dry_run: bool,
) -> None:
    payload = _aggregate_latest_payload_for_model(result_files, latest_only=latest_only)
    model_label = _model_label_from_dir(model_dir.name)
    run_name = model_label
    run_id = _deterministic_run_id(model_label)

    latest_result = sorted(result_files, key=_result_sort_key)[-1]
    resolved = payload.get("config", {}).get("resolved_cli_args", {})
    existing_wandb_args = resolved.get("wandb_args", "")
    merged_wandb_args = _merge_wandb_args(
        existing_wandb_args=existing_wandb_args,
        run_name=run_name,
        run_id=run_id,
        project=project,
        entity=entity,
        job_type=job_type,
        resume=resume,
        group=group_override,
    )

    source_file_count = 1 if latest_only else len(result_files)
    mode = "latest-only" if latest_only else "aggregate"
    print(f"[INFO] model={model_label} mode={mode} files={source_file_count} latest_file={latest_result.name}")
    print(f"[INFO] wandb_args={merged_wandb_args}")
    if dry_run:
        return
    if wandb is None:
        raise ModuleNotFoundError("wandb is not installed in this environment; install wandb or use --dry-run.")

    run = wandb.init(**_simple_parse_args_string(merged_wandb_args))
    _log_results_like_lmms_eval(run=run, payload=payload)
    run.summary["lmms_eval/model_dir"] = str(model_dir)
    run.summary["lmms_eval/source_file_count"] = source_file_count
    run.summary["lmms_eval/latest_results_json_name"] = latest_result.name
    run.finish()


def _run_smoke_tests() -> int:
    print("[SMOKE] Running smoke tests...")

    model_label = "ap1p5-8b-256k-example"
    run_id = _deterministic_run_id("seed-example")
    merged = _merge_wandb_args(
        existing_wandb_args="project=lmms-eval,entity=rkreft-eth-z-rich,resume=allow,group=mygroup",
        run_name=model_label,
        run_id=run_id,
        project=DEFAULT_WANDB_PROJECT,
        entity=DEFAULT_WANDB_ENTITY,
        job_type=DEFAULT_WANDB_JOB_TYPE,
        resume=DEFAULT_WANDB_RESUME,
        group="",
    )
    parsed = _simple_parse_args_string(merged)
    assert parsed["name"] == model_label, "run name must equal model label"
    assert parsed["id"] == run_id, "run id must match provided deterministic id"
    assert parsed["resume"] == "must", "resume mode must be forced to CLI default/setting"
    assert parsed["group"] == "mygroup", "existing group should be preserved"

    merged_a = _merge_wandb_args(
        existing_wandb_args="",
        run_name=model_label,
        run_id=run_id,
        project=DEFAULT_WANDB_PROJECT,
        entity=DEFAULT_WANDB_ENTITY,
        job_type=DEFAULT_WANDB_JOB_TYPE,
        resume=DEFAULT_WANDB_RESUME,
        group="g1",
    )
    merged_b = _merge_wandb_args(
        existing_wandb_args="",
        run_name=model_label,
        run_id=run_id,
        project=DEFAULT_WANDB_PROJECT,
        entity=DEFAULT_WANDB_ENTITY,
        job_type=DEFAULT_WANDB_JOB_TYPE,
        resume=DEFAULT_WANDB_RESUME,
        group="g1",
    )
    parsed_a = _simple_parse_args_string(merged_a)
    parsed_b = _simple_parse_args_string(merged_b)
    assert parsed_a["id"] == parsed_b["id"], "same explicit run id must be stable"
    assert str(parsed_a["id"]).startswith("model_"), "run id must match launcher prefix"

    print("[SMOKE] OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Push lmms-eval results JSON files to W&B runs.")
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT, help=f"Root results directory. Default: {DEFAULT_RESULTS_ROOT}")
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT, help=f"W&B project. Default: {DEFAULT_WANDB_PROJECT}")
    parser.add_argument("--wandb-entity", default=DEFAULT_WANDB_ENTITY, help=f"W&B entity/team. Default: {DEFAULT_WANDB_ENTITY}")
    parser.add_argument("--wandb-job-type", default=DEFAULT_WANDB_JOB_TYPE, help=f"W&B job_type. Default: {DEFAULT_WANDB_JOB_TYPE}")
    parser.add_argument("--wandb-resume", default=DEFAULT_WANDB_RESUME, help=f"W&B resume mode. Default: {DEFAULT_WANDB_RESUME}")
    parser.add_argument("--wandb-group", default="", help="Optional default W&B group when missing from results.")
    parser.add_argument("--latest-only", action="store_true", help="Push only the newest *_results.json per model directory.")
    parser.add_argument("--sleep-seconds", type=float, default=5.0, help="Sleep between pushes to reduce W&B 429 rate limits. Default: 5.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved actions without uploading.")
    parser.add_argument("--smoke-test", action="store_true", help="Run local smoke tests and exit.")
    args = parser.parse_args()

    if args.smoke_test:
        return _run_smoke_tests()

    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["WANDB_ENTITY"] = args.wandb_entity

    results_root = Path(args.results_root)
    if not results_root.exists():
        raise FileNotFoundError(f"results root does not exist: {results_root}")

    grouped = _group_results_files_by_model(results_root)
    if not grouped:
        print(f"[WARN] No *_results.json files found under {results_root}")
        return 0

    model_dirs = sorted(grouped.keys(), key=lambda p: p.name)
    for idx, model_dir in enumerate(model_dirs):
        result_files = grouped[model_dir]
        _push_one_model_aggregate(
            model_dir=model_dir,
            result_files=result_files,
            project=args.wandb_project,
            entity=args.wandb_entity,
            job_type=args.wandb_job_type,
            resume=args.wandb_resume,
            group_override=args.wandb_group,
            latest_only=args.latest_only,
            dry_run=args.dry_run,
        )
        if args.sleep_seconds > 0 and idx < len(model_dirs) - 1:
            print(f"[INFO] sleeping {args.sleep_seconds:.1f}s before next push...")
            time.sleep(args.sleep_seconds)

    print(f"[DONE] Processed {len(model_dirs)} model run(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
