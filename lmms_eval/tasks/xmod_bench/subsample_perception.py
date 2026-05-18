"""
Subsample perception/finegrained and perception/general_activities to exactly
1000 clips × 6 modality files = 6,000 samples per subtask.

Algorithm
---------
1. Scan all JSONL files and, for each finegrained/general_activities sample,
   derive a clip ID from the media paths.
2. Keep only clips that appear in ALL 6 relevant modality files.
3. Sample 1000 clip IDs (fixed seed).
4. For each selected clip × each JSONL file: keep exactly ONE question
   (if the clip has multiple questions in that file, pick one at random).
5. Overwrite the JSONL files (other subtasks are left unchanged).

Clip ID derivation (VGGSound data)
-----------------------------------
  Audio correct  → stem of wav filename       (e.g. "NXCTvtrCPXs_000177")
  Image correct  → parent dir name - "_frames" (e.g. "NXCTvtrCPXs_000177")
  Text  correct  → fall back to condition media (Audio or Image path)

Usage
-----
    python subsample_perception.py [--seed 42] [--dry-run]
"""

import argparse
import json
import os
import random
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# JSONL files that contain finegrained / general_activities samples
# (VGGSound is audio + image only → no video-containing files)
TARGET_FILES = [
    "audio_image.jsonl",
    "audio_text.jsonl",
    "image_audio.jsonl",
    "image_text.jsonl",
    "text_audio.jsonl",
    "text_image.jsonl",
]

FINEGRAINED_CATS = {
    "Animal Sounds",
    "Human Activities",
    "Human Speech",
    "Musical Instruments",
    "Natural Sounds",
    "Tools & Machinery",
    "Transportation",
    "Urban Sounds",
}
GENERAL_CATS = {"general_activities"}
ALL_TARGET_CATS = FINEGRAINED_CATS | GENERAL_CATS

N_INSTANCES = 1000


# ---------------------------------------------------------------------------
# Clip ID helpers
# ---------------------------------------------------------------------------


def _audio_stem(path: str) -> str:
    return os.path.basename(path).rsplit(".", 1)[0]


def _image_stem(path: str) -> str:
    return os.path.basename(os.path.dirname(path)).replace("_frames", "")


def clip_id(doc: dict) -> str | None:
    """Return a unique clip ID regardless of which modality is the condition."""
    correct = doc["options"][doc["correct_answer"]]
    if correct["modality"] == "Audio":
        return _audio_stem(correct["input"])
    if correct["modality"] == "Image":
        return _image_stem(correct["input"])
    # correct is Text → fall back to condition
    cond = doc["conditions"]
    if cond["modality"] == "Audio":
        return _audio_stem(cond["input"])
    if cond["modality"] == "Image":
        return _image_stem(cond["input"])
    return None


def subtask_of(cat: str) -> str | None:
    if cat in FINEGRAINED_CATS:
        return "finegrained"
    if cat in GENERAL_CATS:
        return "general_activities"
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    rng = random.Random(args.seed)

    print(f"Seed: {args.seed}  |  dry-run: {args.dry_run}\n")

    # ── Step 1: build clip → {subtask: {file: [doc, ...]}} index ────────────
    # clip_docs[subtask][clip_id][fname] = [doc, ...]
    clip_docs: dict[str, dict[str, dict[str, list]]] = {
        "finegrained": defaultdict(lambda: defaultdict(list)),
        "general_activities": defaultdict(lambda: defaultdict(list)),
    }

    for fname in TARGET_FILES:
        path = os.path.join(DATA_DIR, fname)
        with open(path) as f:
            for line in f:
                doc = json.loads(line)
                cat = doc.get("category", "")
                sub = subtask_of(cat)
                if sub is None:
                    continue
                cid = clip_id(doc)
                if cid:
                    clip_docs[sub][cid][fname].append(doc)

    # ── Step 2: select 1000 clips per subtask ───────────────────────────────
    selected: dict[str, set[str]] = {}
    for sub in ("finegrained", "general_activities"):
        # Only consider clips that appear in ALL 6 target files
        eligible = [cid for cid, files in clip_docs[sub].items() if len(files) == len(TARGET_FILES)]
        print(f"{sub}: {len(clip_docs[sub])} unique clips, " f"{len(eligible)} in all {len(TARGET_FILES)} modality files")
        if len(eligible) < N_INSTANCES:
            raise ValueError(f"Only {len(eligible)} eligible clips for {sub}; " f"need {N_INSTANCES}")
        selected[sub] = set(rng.sample(eligible, N_INSTANCES))
        print(f"  → sampled {N_INSTANCES} clips (seed={args.seed})")

    # ── Step 3: rewrite JSONL files ─────────────────────────────────────────
    print()
    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.endswith(".jsonl"):
            continue
        path = os.path.join(DATA_DIR, fname)

        with open(path) as f:
            all_lines = f.readlines()

        if fname not in TARGET_FILES:
            print(f"  {fname:35s}  (unchanged, {len(all_lines):,} rows)")
            continue

        kept_other = []  # non-target-subtask rows → keep all
        kept_target = []  # one row per (clip, subtask) from this file

        for line in all_lines:
            doc = json.loads(line)
            cat = doc.get("category", "")
            sub = subtask_of(cat)
            if sub is None:
                kept_other.append(line)
                continue
            cid = clip_id(doc)
            if cid in selected.get(sub, set()):
                kept_target.append((sub, cid, line))

        # For each (sub, clip), keep exactly ONE question from this file
        seen: set[tuple[str, str]] = set()
        deduped = []
        # Shuffle so the single kept row isn't always the first occurrence
        rng.shuffle(kept_target)
        for sub, cid, line in kept_target:
            key = (sub, cid)
            if key not in seen:
                seen.add(key)
                deduped.append(line)

        out_lines = kept_other + deduped
        before, after = len(all_lines), len(out_lines)
        print(f"  {fname:35s}  {before:,} → {after:,} rows  " f"({len(deduped):,} target kept)")

        if not args.dry_run:
            with open(path, "w") as f:
                f.writelines(out_lines)

    if args.dry_run:
        print("\n(dry-run: no files written)")
    else:
        print("\nDone.")


if __name__ == "__main__":
    main()
