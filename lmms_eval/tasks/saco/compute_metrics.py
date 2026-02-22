#!/usr/bin/env python3
"""Compute final SaCO / PBench metrics (MCC, pmF1, cgF1) from lmms-eval results JSON.

Usage::

    python -m lmms_eval.tasks.saco.compute_metrics results/facebook__sam3/*_results.json

The script reads the aggregated TP/TN/FP/FN sums and per-threshold
localization counts already present in the JSON, computes the derived
dataset-level metrics, prints a summary table, and writes the metrics
back into the JSON.
"""

import argparse
import sys
from pathlib import Path

from lmms_eval.tasks.saco.utils import compute_saco_final_metrics


def _print_table(summary: dict, json_path: str) -> None:
    """Print a formatted results table similar to run_falcon_eval.py."""
    if not summary:
        print("  No saco_gold / pbench tasks found in results.")
        return

    # Determine benchmark name from task keys
    task_names = sorted(summary.keys())
    is_pbench = any(t.startswith("pbench") for t in task_names)
    is_saco = any(t.startswith("saco_gold") for t in task_names)
    bench_name = "PBench" if is_pbench else "SaCO-Gold" if is_saco else "Segmentation"

    # Strip common prefix for cleaner display
    if is_pbench:
        display = {t: t.replace("pbench_", "") for t in task_names}
    elif is_saco:
        display = {t: t.replace("saco_gold_", "") for t in task_names}
    else:
        display = {t: t for t in task_names}

    # Header
    w_split = max(max(len(v) for v in display.values()), 12)
    hdr = f"{'Split':<{w_split}}" f"  {'IL_MCC':>8}" f"  {'pmF1':>8}" f"  {'cgF1':>8}" f"  {'macro_F1':>8}" f"  {'cnt_acc':>8}"
    sep = "-" * len(hdr)

    print(f"\n{'=' * len(hdr)}")
    print(f"{bench_name} RESULTS — {json_path}")
    print(f"{'=' * len(hdr)}")
    print(hdr)
    print(sep)

    # Per-split rows
    totals = {"IL_MCC": 0, "pmF1": 0, "cgF1": 0, "macro_F1": 0, "cnt_acc": 0}
    n = 0
    for t in task_names:
        m = summary[t]
        label = display[t]

        # count_accuracy comes from the original JSON (not computed here)
        cnt_acc = m.get("count_accuracy", 0.0)

        print(f"{label:<{w_split}}" f"  {m['IL_MCC']:>8.4f}" f"  {m['pmF1'] * 100:>7.1f}%" f"  {m['cgF1']:>7.2f}" f"  {m['macro_F1'] * 100:>7.1f}%" f"  {cnt_acc * 100:>7.1f}%")
        totals["IL_MCC"] += m["IL_MCC"]
        totals["pmF1"] += m["pmF1"]
        totals["cgF1"] += m["cgF1"]
        totals["macro_F1"] += m["macro_F1"]
        totals["cnt_acc"] += cnt_acc
        n += 1

    if n > 1:
        print(sep)
        print(f"{'AVERAGE':<{w_split}}" f"  {totals['IL_MCC'] / n:>8.4f}" f"  {totals['pmF1'] / n * 100:>7.1f}%" f"  {totals['cgF1'] / n:>7.2f}" f"  {totals['macro_F1'] / n * 100:>7.1f}%" f"  {totals['cnt_acc'] / n * 100:>7.1f}%")
    print(f"{'=' * len(hdr)}\n")


def main():
    parser = argparse.ArgumentParser(description="Compute SaCO / PBench final metrics (MCC, pmF1, cgF1) from lmms-eval results JSON.")
    parser.add_argument(
        "results_json",
        type=str,
        nargs="+",
        help="Path(s) to lmms-eval *_results.json file(s).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write derived metrics back into the JSON.",
    )
    args = parser.parse_args()

    for path in args.results_json:
        if not Path(path).exists():
            print(f"ERROR: File not found: {path}", file=sys.stderr)
            continue

        summary = compute_saco_final_metrics(path, save=not args.no_save)
        _print_table(summary, path)

        if not args.no_save:
            print(f"  Derived metrics written back to {path}")


if __name__ == "__main__":
    main()
