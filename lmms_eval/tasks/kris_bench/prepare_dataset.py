#!/usr/bin/env python3
"""
Prepare KRIS-Bench index file for lmms-eval.

This creates a small jsonl index (no images embedded) that lmms-eval can load
with `dataset_path: json`.

Default assumes the benchmark is available at:
  /pfs/training-data/kemingwu/workspace/github_repo/Kris_Bench/KRIS_Bench

Usage:
  python lmms_eval/tasks/kris_bench/prepare_dataset.py \
    --bench_root /path/to/KRIS_Bench \
    --output_jsonl lmms_eval/tasks/kris_bench/kris_bench.jsonl
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Build KRIS-Bench jsonl index for lmms-eval")
    parser.add_argument(
        "--bench_root",
        type=str,
        default="/pfs/training-data/kemingwu/workspace/github_repo/Kris_Bench/KRIS_Bench",
        help="Path to KRIS_Bench directory containing category subfolders",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="/pfs/training-data/kemingwu/workspace/unify_model/lmms-eval/lmms_eval/tasks/kris_bench/kris_bench.jsonl",
        help="Output jsonl path",
    )
    args = parser.parse_args()

    bench_root = Path(args.bench_root)
    out_path = Path(args.output_jsonl)
    ann_paths = sorted(bench_root.glob("*/annotation.json"))
    if not ann_paths:
        raise FileNotFoundError(f"No annotation.json found under {bench_root}")

    rows = []
    for ann_path in ann_paths:
        category = ann_path.parent.name
        data = json.load(open(ann_path, "r", encoding="utf-8"))
        for image_id, entry in data.items():
            ori = entry.get("ori_img", [])
            if isinstance(ori, str):
                ori = [ori]
            rows.append(
                {
                    "key": f"{category}__{image_id}",
                    "category": category,
                    "image_id": str(image_id),
                    "ori_img": ori,
                    "gt_img": entry.get("gt_img", "") or "",
                    "ins_en": entry.get("ins_en", "") or "",
                    "explain_en": entry.get("explain_en", "") or "",
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} samples to {out_path}")


if __name__ == "__main__":
    main()
