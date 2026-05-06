#!/usr/bin/env python3
"""
hd_epic_to_hf.py
================
Convert HD-EPIC question JSON files into a single JSONL that lmms-eval can
load via `dataset_path: json`. Optionally pushes to the HuggingFace Hub.

The official HD-EPIC annotation format (hd-epic-annotations/vqa-benchmark/)
uses pre-shuffled choices with a correct_idx pointer:

  {
    "<question_id>": {
      "question":    "...",
      "choices":     ["option A", "option B", ...],
      "correct_idx": 2,
      "inputs": {
        "video 1": {"id": "P01-...", "start_time": "...", "end_time": "..."},
        "image 1": {"id": "P01-...", "time": "..."}
      }
    }
  }

A fallback branch also handles a hypothetical correct + incorrect layout
in case you point the script at a different annotation source, but this is
not present in the official HD-EPIC data.

Usage
-----
  python hd_epic_to_hf.py \\
      --questions-dir /path/to/hd-epic-annotations/vqa-benchmark \\
      --output         hd_epic_questions.jsonl \\
      --video-dir      /path/to/videos                    # optional

  # Push to HuggingFace Hub:
  python hd_epic_to_hf.py ... --push-to-hub your-org/hd-epic
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import os.path as osp
import random
import sys
from typing import Iterator


def _secs_from_time_str(s: str) -> float:
    """
    Parse 'HH:MM:SS.fff' (or 'HH:MM:SS', or short un-padded variants) ->
    float seconds. Returns -1.0 on any failure rather than raising, so a
    single malformed timestamp never aborts a full conversion run.
    Kept in sync with the identical function in utils.py.
    """
    try:
        h_str, m_str, s_str = s.split(":", 2)
        return int(h_str) * 3600 + int(m_str) * 60 + float(s_str)
    except (ValueError, AttributeError, TypeError):
        return -1.0


def _build_choices(q: dict, rng: random.Random) -> tuple[list, int]:
    """
    Return (choices, correct_idx).

    - If `choices` + `correct_idx` already present, use as-is.
    - Else if `correct` + `incorrect` present, build choices by inserting
      the correct answer at a deterministic random position.
    """
    if "choices" in q and "correct_idx" in q:
        return q["choices"], int(q["correct_idx"])

    if "correct" in q and "incorrect" in q:
        correct = q["correct"]
        incorrect = list(q["incorrect"])
        # Insert correct answer at a random position rather than appending
        # then shuffling -- avoids the .index() ambiguity if the correct
        # answer text happens to appear in the incorrect list too.
        insert_pos = rng.randint(0, len(incorrect))
        all_opts = incorrect[:insert_pos] + [correct] + incorrect[insert_pos:]
        return all_opts, insert_pos

    raise KeyError(f"Question {q.get('question_id', '?')} has neither (choices, correct_idx) " f"nor (correct, incorrect) -- cannot build MCQ.")


def _iter_question_file(path: str, seed: int) -> Iterator[dict]:
    """Yield one record per question from a single JSON file."""
    task_type = osp.splitext(osp.basename(path))[0]

    with open(path, "r") as f:
        data = json.load(f)

    # Stable per-file RNG so shuffling is reproducible across runs.
    rng = random.Random(f"{seed}:{task_type}")

    for q_id, q in data.items():
        try:
            choices, correct_idx = _build_choices(q, rng)
        except KeyError as exc:
            print(f"  warn: skipping {q_id} ({exc})", file=sys.stderr)
            continue

        inputs: dict = q.get("inputs", {})
        video_ids: list = []
        input_labels: list = []
        start_times: list = []
        end_times: list = []

        for label, inp in inputs.items():
            video_ids.append(inp["id"])
            input_labels.append(label)

            if "image" in label:
                # Single-frame reference -> 1-second clip
                t = _secs_from_time_str(inp.get("time", "00:00:00.000"))
                start_times.append(t)
                end_times.append(t + 1.0)
            else:
                st = _secs_from_time_str(inp["start_time"]) if "start_time" in inp else -1.0
                et = _secs_from_time_str(inp["end_time"]) if "end_time" in inp else -1.0
                start_times.append(st)
                end_times.append(et)

        yield {
            "question_id": q_id,
            "task_type": task_type,
            "question": q.get("question", ""),
            "choices": choices,
            "correct_idx": correct_idx,
            "video_ids": video_ids,
            "input_labels": input_labels,
            "start_times": start_times,
            "end_times": end_times,
        }


def convert(
    questions_dir: str,
    output_path: str,
    video_dir: str = "",
    seed: int = 42,
    glob_pat: str = "*.json",
) -> int:
    pattern = osp.join(questions_dir, glob_pat)
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"ERROR: no files matched {pattern}", file=sys.stderr)
        return 0

    n = 0
    with open(output_path, "w") as out_f:
        for fp in files:
            print(f"  {osp.basename(fp)}", end=" ", flush=True)
            count = 0
            for record in _iter_question_file(fp, seed):
                record["video_dir"] = video_dir
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
                n += 1
            print(f"-> {count} questions")

    print(f"\nWrote {n} questions to {output_path}")
    return n


def push_to_hub(jsonl_path: str, hub_dataset_id: str):
    try:
        from datasets import Dataset
    except ImportError:
        print("ERROR: pip install datasets", file=sys.stderr)
        sys.exit(1)
    print(f"Loading JSONL...")
    ds = Dataset.from_json(jsonl_path)
    print(f"Pushing {len(ds)} records to {hub_dataset_id}...")
    ds.push_to_hub(hub_dataset_id, split="test")
    print("Done.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--questions-dir", required=True, help="Directory of per-prototype JSON files")
    p.add_argument("--output", default="hd_epic_questions.jsonl")
    p.add_argument("--video-dir", default="", help="Base directory for video .mp4 files (overridable at runtime via $HD_EPIC_VIDEO_DIR)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for choice shuffling (default: 42)")
    p.add_argument("--glob", default="*.json")
    p.add_argument("--push-to-hub", metavar="DATASET_ID", default=None)
    args = p.parse_args()

    n = convert(args.questions_dir, args.output, args.video_dir, args.seed, args.glob)
    if n == 0:
        sys.exit(1)
    if args.push_to_hub:
        push_to_hub(args.output, args.push_to_hub)


if __name__ == "__main__":
    main()
