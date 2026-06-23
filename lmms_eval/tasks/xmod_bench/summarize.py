"""summarize.py — produce XModBench Level-2 metrics from lmms-eval logs.

For each XModBench task run (full or lite), lmms-eval writes a per-sample log
when invoked with `--log_samples`. Each sample carries the per-doc result dict
`xmod_bench_score: {family, subtask, config, correct}` that we emit in
`utils.xmod_bench_process_results`. This script ingests one or more of those
logs, computes the canonical Level-2 numbers reported in the paper, and prints
them as a table + JSON.

Level-2 (17 numbers total):
  1. By config (6):  mean over (family, subtask) cells of acc[(f,s)][config]
  2. By family (5):  for each family, mean over 6 configs of
                     (mean over subtasks in family of acc[subtask][config])
  3. Modality disparity (3):
        ΔT_vs_V = (Acc[a2v] − Acc[a2t]) + (Acc[v2a] − Acc[t2a])
        ΔT_vs_A = (Acc[v2a] − Acc[v2t]) + (Acc[a2v] − Acc[t2v])
        ΔV_vs_A = (Acc[t2a] − Acc[t2v]) + (Acc[a2t] − Acc[v2t])
  4. Directional imbalance (3):
        Δ_T↔V = Acc[t2v] − Acc[v2t]
        Δ_T↔A = Acc[t2a] − Acc[a2t]
        Δ_V↔A = Acc[v2a] − Acc[a2v]

Usage:
    python summarize.py --logs <output_path>/<run-id>/ [--out summary.json]

The `--logs` path can be either a directory containing the per-task
`samples_<task>_*.jsonl` files, or a glob pattern matching such files.
"""

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

FAMILIES = ["perception", "spatial", "temporal", "linguistic", "knowledge"]
CONFIGS = ["a2t", "a2v", "t2a", "t2v", "v2a", "v2t"]


def find_sample_files(spec: str) -> list[Path]:
    p = Path(spec)
    if p.is_dir():
        return sorted(p.rglob("*xmod_bench*.jsonl")) + sorted(p.rglob("*xmod_bench*samples*.json"))
    return [Path(x) for x in sorted(glob.glob(spec))]


def iter_samples(path: Path):
    """Yield each sample record from a lmms-eval samples file (jsonl or json list)."""
    text = path.read_text()
    text = text.strip()
    if not text:
        return
    if text.lstrip().startswith("["):
        for r in json.loads(text):
            yield r
        return
    for line in text.splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)


def extract_record(sample: dict) -> dict | None:
    """Pull the xmod_bench_score dict out of a lmms-eval sample row."""
    score = sample.get("xmod_bench_score")
    if isinstance(score, dict) and "family" in score:
        return score
    if isinstance(sample.get("doc"), dict) and "xmod_bench_score" in sample:
        s = sample["xmod_bench_score"]
        if isinstance(s, dict) and "family" in s:
            return s
    return None


def aggregate(paths: list[Path]) -> dict:
    # Counts keyed by ((family, subtask), config)
    correct: dict[tuple[tuple[str, str], str], int] = defaultdict(int)
    total: dict[tuple[tuple[str, str], str], int] = defaultdict(int)
    for p in paths:
        n = 0
        for sample in iter_samples(p):
            rec = extract_record(sample)
            if rec is None:
                continue
            cell = ((rec["family"], rec["subtask"]), rec["config"])
            correct[cell] += int(rec["correct"])
            total[cell] += 1
            n += 1
        print(f"  loaded {n:>6d} samples from {p.name}")

    def acc(c, t):
        return 100.0 * c / t if t else float("nan")

    # acc[cell][config]
    cells = sorted({k[0] for k in total})
    acc_cell_cfg: dict[tuple[str, str], dict[str, float]] = {}
    for cell in cells:
        per_cfg = {}
        for cfg in CONFIGS:
            per_cfg[cfg] = acc(correct.get((cell, cfg), 0), total.get((cell, cfg), 0))
        acc_cell_cfg[cell] = per_cfg

    # By config (mean over all (family, subtask) cells present for this config)
    by_config: dict[str, float] = {}
    for cfg in CONFIGS:
        vals = [acc_cell_cfg[cell][cfg] for cell in cells if total.get((cell, cfg), 0) > 0]
        by_config[cfg] = sum(vals) / len(vals) if vals else float("nan")

    # By family: for each family, average across 6 configs of (mean acc over subtasks in family)
    by_family: dict[str, float] = {}
    for fam in FAMILIES:
        fam_cells = [c for c in cells if c[0] == fam]
        if not fam_cells:
            by_family[fam] = float("nan")
            continue
        cfg_vals = []
        for cfg in CONFIGS:
            sub_vals = [acc_cell_cfg[c][cfg] for c in fam_cells if total.get((c, cfg), 0) > 0]
            if sub_vals:
                cfg_vals.append(sum(sub_vals) / len(sub_vals))
        by_family[fam] = sum(cfg_vals) / len(cfg_vals) if cfg_vals else float("nan")

    a = by_config  # alias for readability
    disparity = {
        "T_vs_V": (a["a2v"] - a["a2t"]) + (a["v2a"] - a["t2a"]),
        "T_vs_A": (a["v2a"] - a["v2t"]) + (a["a2v"] - a["t2v"]),
        "V_vs_A": (a["t2a"] - a["t2v"]) + (a["a2t"] - a["v2t"]),
    }
    imbalance = {
        "T_vs_V": a["t2v"] - a["v2t"],
        "T_vs_A": a["t2a"] - a["a2t"],
        "V_vs_A": a["v2a"] - a["a2v"],
    }

    return {
        "by_config": by_config,
        "by_family": by_family,
        "disparity": disparity,
        "imbalance": imbalance,
        "level1": {f"{f}/{s}": acc_cell_cfg[(f, s)] for (f, s) in cells},
    }


def print_summary(s: dict) -> None:
    print("\n=== Level-2 summary ===\n")
    print("By config:")
    for cfg in CONFIGS:
        print(f"  {cfg:6s}: {s['by_config'][cfg]:6.2f}")
    print("\nBy family:")
    for fam in FAMILIES:
        print(f"  {fam:12s}: {s['by_family'][fam]:6.2f}")
    print("\nModality disparity:")
    for k, v in s["disparity"].items():
        print(f"  Δ_{k}: {v:+6.2f}")
    print("\nDirectional imbalance:")
    for k, v in s["imbalance"].items():
        print(f"  Δ_{k}: {v:+6.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, help="dir or glob with samples_*.jsonl from lmms-eval")
    ap.add_argument("--out", help="optional JSON output path")
    args = ap.parse_args()

    paths = find_sample_files(args.logs)
    if not paths:
        raise SystemExit(f"no sample files matched: {args.logs}")
    print(f"found {len(paths)} sample files")

    summary = aggregate(paths)
    print_summary(summary)

    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
