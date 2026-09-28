#!/usr/bin/env python3
"""Shrink a finished run dir for git (stdlib only; run_eval.sh calls it after every run, idempotent).

- run.log / server.log: progress bars rewrite one line with carriage returns; keep only each line's final state.
- tegrastats.log, lmms_eval/*_samples_*.jsonl -> gzipped (summarize.py and flops.py read .gz transparently).
- flops.json: per-sample rows move to flops_samples.jsonl.gz; flops.json keeps the per-task summary.

Usage: python3 jetson/scripts/slim_run.py <run dir> [<run dir> ...]
"""

import glob
import gzip
import json
import os
import re
import shutil
import sys

# Per-request noise: vLLM's inner progress bars and Edge-LLM's profile switch, one line (or two) per sample.
NOISE = re.compile(rb"^(Processed prompts: |Rendering conversations: |.*\[TensorRT\] Switching optimization profile)")
PROGRESS = re.compile(rb"^(.*?:\s+\d+)%\|")  # tqdm bar: "<desc>:  42%|" -> keep one line per desc and percent


def strip_progress(path):
    with open(path, "rb") as f:
        data = f.read()
    lines, dropped, last_bar = [], 0, {}
    for line in data.split(b"\n"):
        line = line.rsplit(b"\r", 1)[-1]
        if NOISE.match(line):
            dropped += 1
            continue
        if bar := PROGRESS.match(line):
            desc, percent = bar.group(1).rsplit(b":", 1)
            if last_bar.get(desc) == percent.strip():
                dropped += 1
                continue
            last_bar[desc] = percent.strip()
        lines.append(line)
    if dropped:
        lines.append(f"[slim_run.py: dropped {dropped} per-request progress/profile lines]".encode())
    slim = b"\n".join(lines)
    if len(slim) < len(data):
        with open(path, "wb") as f:
            f.write(slim)


def gzip_file(path):
    with open(path, "rb") as src, gzip.open(path + ".gz", "wb", compresslevel=9) as dst:
        shutil.copyfileobj(src, dst)
    os.remove(path)


def split_flops(path):
    flops = json.load(open(path))
    rows = [{"task": task, **row} for task, data in flops["tasks"].items() for row in data.pop("samples", [])]
    if rows:
        with gzip.open(os.path.join(os.path.dirname(path), "flops_samples.jsonl.gz"), "wt", compresslevel=9) as f:
            f.writelines(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)
    json.dump(flops, open(path, "w"), indent=1)


def slim(run_dir):
    for name in ("run.log", "server.log", "flops.log"):
        if os.path.exists(os.path.join(run_dir, name)):
            strip_progress(os.path.join(run_dir, name))
    for path in [os.path.join(run_dir, "tegrastats.log"), *glob.glob(os.path.join(run_dir, "lmms_eval", "*_samples_*.jsonl"))]:
        if os.path.exists(path):
            gzip_file(path)
    if os.path.exists(os.path.join(run_dir, "flops.json")):
        split_flops(os.path.join(run_dir, "flops.json"))


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for run_dir in sys.argv[1:]:
        slim(run_dir)


if __name__ == "__main__":
    main()
