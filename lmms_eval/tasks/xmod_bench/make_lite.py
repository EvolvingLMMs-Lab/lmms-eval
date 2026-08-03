"""make_lite.py — build XModBench-Lite (6k samples).

Lite spec (paper Sec 3):
  5 task families x 6 modality configurations x 200 examples per (family, config).
  Vision = image U video. Each config yields one Lite JSONL with 1000 rows.

Run:
    python make_lite.py \\
        --src /scratch/xwang378/2025/lmms-eval/lmms_eval/tasks/xmod_bench/data \\
        --dst /home/xwang378/scratch/2025/AudioBench_data/data_lite \\
        --seed 42
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

# Maps family-prefix -> short canonical name used in the paper.
FAMILY_OF = {
    "01_perception": "perception",
    "02_spatial": "spatial",
    "03_speech": "linguistic",
    "04_temporal": "temporal",
    "05_Exteral": "knowledge",  # source data has this typo
    "05_exteral": "knowledge",
    "05_external": "knowledge",
}

FAMILIES = ["perception", "spatial", "temporal", "linguistic", "knowledge"]

# (config_name, [source modality combos that map to this config])
CONFIGS = [
    ("a2t", ["audio_text"]),
    ("a2v", ["audio_image", "audio_video"]),
    ("t2a", ["text_audio"]),
    ("t2v", ["text_image", "text_video"]),
    ("v2a", ["image_audio", "video_audio"]),
    ("v2t", ["image_text", "video_text"]),
]

PER_CELL = 200  # samples per (family, config) cell


def family_of(subtask: str) -> str:
    prefix = subtask.split("/")[0] if "/" in subtask else subtask
    return FAMILY_OF.get(prefix.lower(), FAMILY_OF.get(prefix, "other"))


def load_combo(src: Path, combo: str) -> list[dict]:
    rows = []
    path = src / f"{combo}.jsonl"
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.dst.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    summary: dict[tuple[str, str], int] = {}
    for config_name, source_combos in CONFIGS:
        # Bucket source rows by (family, subtask)
        family_to_subtask_rows: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
        for combo in source_combos:
            for row in load_combo(args.src, combo):
                fam = family_of(row["subtask"])
                family_to_subtask_rows[fam][row["subtask"]].append(row)

        # Sample 200 per family, stratified across the subtasks in that family.
        config_rows: list[dict] = []
        for fam in FAMILIES:
            subtasks = family_to_subtask_rows.get(fam, {})
            if not subtasks:
                summary[(fam, config_name)] = 0
                continue
            picked = stratified_sample(subtasks, PER_CELL, rng)
            summary[(fam, config_name)] = len(picked)
            config_rows.extend(picked)

        rng.shuffle(config_rows)
        out_path = args.dst / f"{config_name}.jsonl"
        with open(out_path, "w") as f:
            for i, row in enumerate(config_rows):
                row["index"] = i
                f.write(json.dumps(row) + "\n")
        print(f"wrote {out_path}  rows={len(config_rows)}")

    print()
    print("=== Lite cell sizes ===")
    print(f"{'family':12s} " + " ".join(f"{c:>5s}" for c, _ in CONFIGS))
    for fam in FAMILIES:
        row = " ".join(f"{summary.get((fam, c), 0):>5d}" for c, _ in CONFIGS)
        print(f"{fam:12s} {row}")


def stratified_sample(subtasks: dict[str, list[dict]], target: int, rng: random.Random) -> list[dict]:
    """Sample `target` rows total, balancing across the given subtasks.

    Smaller subtasks contribute all their rows; remaining quota is filled from
    larger subtasks. If pool < target, returns whatever is available.
    """
    pool_sizes = {st: len(rows) for st, rows in subtasks.items()}
    total = sum(pool_sizes.values())
    if total <= target:
        out = []
        for rows in subtasks.values():
            out.extend(rows)
        return out

    n_subtasks = len(subtasks)
    per_subtask = target // n_subtasks
    leftover = target - per_subtask * n_subtasks

    quotas: dict[str, int] = {}
    overflow = 0
    for st, n in pool_sizes.items():
        if n <= per_subtask:
            quotas[st] = n
            overflow += per_subtask - n
        else:
            quotas[st] = per_subtask
    quotas_with_room = [st for st, n in pool_sizes.items() if n > quotas[st]]
    extra_needed = overflow + leftover
    while extra_needed > 0 and quotas_with_room:
        per = max(1, extra_needed // len(quotas_with_room))
        new_room = []
        for st in quotas_with_room:
            add = min(per, pool_sizes[st] - quotas[st], extra_needed)
            quotas[st] += add
            extra_needed -= add
            if pool_sizes[st] - quotas[st] > 0 and extra_needed > 0:
                new_room.append(st)
            if extra_needed <= 0:
                break
        quotas_with_room = new_room

    picked: list[dict] = []
    for st, q in quotas.items():
        if q <= 0:
            continue
        picked.extend(rng.sample(subtasks[st], q))
    return picked


if __name__ == "__main__":
    main()
