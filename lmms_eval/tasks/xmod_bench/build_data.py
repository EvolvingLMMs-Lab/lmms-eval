"""
build_data.py — Preprocess AudioBench raw JSON files into XModBench JSONL datasets.

Usage:
    python build_data.py [--tasks-root PATH] [--out-dir PATH] [--seed INT]

Each output JSONL file covers one modality combination (e.g. audio_image.jsonl).
Rows are sorted by (subtask, original_index) and carry an `index` field that
records the row's original position in the source JSON file, so results can be
traced back to the raw data.

Capped subtasks (resample to N instances, same indices applied to all 6 modality
files so cross-modal correspondence is preserved):
    - 01_perception/finegrained        → 1000 instances
    - 01_perception/general_activities → 1000 instances

Mismatch subtasks (files have different row counts across modality combinations):
    - Detected automatically via conditions.input intersection within each
      condition-modality group (e.g. all "audio" condition files for instruments).
    - The union of skip positions is applied to ALL modality files so positional
      alignment is preserved across the 6 combinations.
    - Known case: 01_perception/instruments — audio_text has 999 rows (missing
      Cello/1u3yHICR_BU.wav at orig_idx 308), all other files have 1000 rows.
      Fix: skip position 308 from every 1000-row file; remap orig_idx for audio_text.
"""

import argparse
import json
import os
import random
from collections import defaultdict

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_TASKS_ROOT = "/home/xwang378/scratch/2025/AudioBench/benchmark/tasks"
DEFAULT_OUT_DIR = "/home/xwang378/scratch/2025/lmms-eval/lmms_eval/tasks/xmod_bench/data"
DEFAULT_SEED = 42

# Subtasks capped at N instances.  All modality files for the same subtask
# share the same sampled index set to preserve cross-modal correspondence.
CAPPED_SUBTASKS: dict[str, int] = {
    "01_perception/finegrained": 1000,
    "01_perception/general_activities": 1000,
    "01_perception/natures": 500,  # 500/file × 12 files = 6000 total, 1000/combo
}

# Subtasks where files with more than N rows are truncated at row N (last rows dropped).
# Used when a few modality files have slightly more rows than others due to source
# data differences, and we simply want to align them at the shorter count.
TRUNCATE_AT: dict[str, int] = {
    "02_spatial/panaroma": 390,  # text_audio/video_audio/video_text have 395; drop last 5
}

# Subtasks excluded from automatic mismatch detection.
MISMATCH_SKIP_SUBTASKS: set[str] = set()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def normalize_media(m: dict) -> dict:
    """Keep only 'modality' and 'input'; drop any extra metadata keys."""
    return {"modality": m.get("modality", ""), "input": m.get("input", "")}


def _compute_mismatch_info(
    subtask: str,
    file_list: list[tuple[str, list]],
) -> dict[str, dict] | None:
    """For a subtask where files have different row counts, compute per-file
    filtering info so that only instances present in ALL files are kept.

    Strategy:
      1. Group files by condition modality (e.g. "audio", "image", "video").
      2. For each group that has count mismatches, intersect conditions.input
         across files → identifies which condition inputs are universally present.
      3. From the full-length files in that group, record the file positions that
         are NOT in the intersection → these are the global_skip_positions.
      4. Apply global_skip_positions to ALL files in the subtask:
           - Full-length files: skip those positions.
           - Short files: all their rows are valid; remap orig_idx so it lines up
             with the full-length file's position numbering.

    Returns a dict: path → {"skip_set": set[int], "canonical_skips": list[int]}
      - skip_set non-empty  → full-length file; skip these file positions.
      - canonical_skips non-empty → short file; remap orig_idx using these offsets.
    Returns None if all files have the same row count (no action needed).
    """
    n_per_file = [len(d) for _, d in file_list]
    if len(set(n_per_file)) == 1:
        return None  # all same → nothing to do

    min_n = min(n_per_file)

    # Group by condition modality
    by_cond_mod: dict[str, list[tuple[str, list]]] = defaultdict(list)
    for path, data in file_list:
        cond_mod = data[0]["conditions"]["modality"].lower()
        by_cond_mod[cond_mod].append((path, data))

    # For each condition-modality group that has mismatched counts,
    # find the intersection of conditions.input and derive skip positions.
    global_skip_positions: set[int] = set()

    for cond_mod, files in by_cond_mod.items():
        counts = [len(d) for _, d in files]
        if len(set(counts)) == 1:
            continue  # uniform count within this group → skip

        # Intersect conditions.input across all files in the group
        sets = [set(item["conditions"]["input"] for item in data) for _, data in files]
        valid = sets[0]
        for s in sets[1:]:
            valid &= s

        # Find positions to skip from the first full-length file in the group
        for path, data in files:
            if len(data) > len(valid):
                for i, item in enumerate(data):
                    if item["conditions"]["input"] not in valid:
                        global_skip_positions.add(i)
                break  # one full-length file per group is sufficient

    if not global_skip_positions:
        # Count mismatch exists but our intersection didn't find skip positions
        # (shouldn't happen with valid data; log and skip)
        print(f"[MISMATCH] WARNING: {subtask} has unequal row counts " f"{set(n_per_file)} but no skip positions found — skipping fix")
        return None

    global_skip_sorted = sorted(global_skip_positions)
    n_valid = max(n_per_file) - len(global_skip_positions)
    print(f"[MISMATCH] {subtask}: row counts {sorted(set(n_per_file))} " f"→ keeping {n_valid} instances  (skip positions: {global_skip_sorted})")

    # Build per-file filtering info
    path_info: dict[str, dict] = {}
    for path, data in file_list:
        if len(data) == min_n:
            # Short file: rows are already a subset of the full sequence.
            # All file positions are valid; orig_idx must be remapped.
            path_info[path] = {"skip_set": set(), "canonical_skips": global_skip_sorted}
        else:
            # Full-length file: skip the positions identified above.
            path_info[path] = {"skip_set": global_skip_positions, "canonical_skips": []}

    return path_info


def _remap_orig_idx(file_pos: int, canonical_skips: list[int]) -> int:
    """Remap a file position in a short file to orig_idx in the full sequence.

    canonical_skips: sorted list of positions that were removed from the
    full-length sequence (e.g. [308] for instruments/audio_text).

    For each skip position s (in order), if the current orig_idx >= s, the
    full sequence has a gap there, so we shift orig_idx up by 1.
    """
    orig_idx = file_pos
    for skip in canonical_skips:  # must be sorted ascending
        if orig_idx >= skip:
            orig_idx += 1
        # No early break: after incrementing, orig_idx might reach the next skip
    return orig_idx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build(tasks_root: str, out_dir: str, seed: int) -> None:
    # ── Step 1: walk task directory, group files by subtask ────────────────
    subtask_files: dict[str, list[tuple[str, list]]] = {}

    for dirpath, _, files in os.walk(tasks_root):
        for fname in files:
            if not fname.endswith(".json"):
                continue
            path = os.path.join(dirpath, fname)
            with open(path) as fp:
                data = json.load(fp)
            if not isinstance(data, list) or not data:
                continue
            d = data[0]
            if not isinstance(d, dict) or not isinstance(d.get("conditions"), dict):
                continue
            subtask = os.path.relpath(dirpath, tasks_root)
            subtask_files.setdefault(subtask, []).append((path, data))

    # ── Step 2: sample shared indices for capped subtasks ──────────────────
    sampled_indices: dict[str, set[int]] = {}
    for subtask, cap in CAPPED_SUBTASKS.items():
        if subtask not in subtask_files:
            print(f"WARNING: capped subtask {subtask!r} not found in {tasks_root}")
            continue
        n_per_file = [len(d) for _, d in subtask_files[subtask]]
        assert len(set(n_per_file)) == 1, f"Row-count mismatch across modality files for {subtask}: {n_per_file}"
        n = n_per_file[0]
        rng = random.Random(seed)
        chosen = set(rng.sample(range(n), cap))
        sampled_indices[subtask] = chosen
        print(f"[CAP] {subtask}: sample {cap}/{n}  (seed={seed})")

    # ── Step 2.5: detect + handle count mismatches in non-capped subtasks ──
    # For any subtask where the modality files have different row counts,
    # compute per-file skip sets and orig_idx remapping so only the N instances
    # present in ALL files are kept (N = intersection size).
    mismatch_info: dict[str, dict[str, dict]] = {}
    for subtask, file_list in subtask_files.items():
        if subtask in CAPPED_SUBTASKS or subtask in MISMATCH_SKIP_SUBTASKS or subtask in TRUNCATE_AT:
            continue
        info = _compute_mismatch_info(subtask, file_list)
        if info is not None:
            mismatch_info[subtask] = info

    # ── Step 3: build rows per modality combo ──────────────────────────────
    combos: dict[str, list[dict]] = defaultdict(list)

    for subtask, file_list in subtask_files.items():
        keep = sampled_indices.get(subtask)  # None → keep all rows
        file_mismatch = mismatch_info.get(subtask)  # None → no mismatch
        truncate = TRUNCATE_AT.get(subtask)  # None → no truncation

        for path, data in file_list:
            d0 = data[0]
            cond = d0.get("conditions", {})
            opts = d0.get("options", {})
            if not isinstance(cond, dict) or not isinstance(opts, dict) or not opts:
                continue
            first_opt = list(opts.values())[0]
            if not isinstance(first_opt, dict):
                continue

            cond_mod = cond.get("modality", "?").lower()
            opt_mod = first_opt.get("modality", "?").lower()
            combo = f"{cond_mod}_{opt_mod}"

            # Per-file mismatch filtering info (None if no mismatch)
            file_info = file_mismatch.get(path) if file_mismatch else None

            for file_pos, item in enumerate(data):
                if not isinstance(item, dict):
                    continue

                # ── Truncation (drop tail rows beyond limit) ───────────────
                if truncate is not None and file_pos >= truncate:
                    break

                # ── Mismatch filtering ─────────────────────────────────────
                if file_info is not None:
                    if file_pos in file_info["skip_set"]:
                        continue  # bad position → skip
                    if file_info["canonical_skips"]:
                        orig_idx = _remap_orig_idx(file_pos, file_info["canonical_skips"])
                    else:
                        orig_idx = file_pos
                else:
                    orig_idx = file_pos

                # ── Capped-subtask index filter ────────────────────────────
                if keep is not None and orig_idx not in keep:
                    continue

                raw_cond = item.get("conditions", {})
                raw_opts = item.get("options", {})
                row = {
                    "index": orig_idx,  # original row index in source JSON
                    "subtask": subtask,
                    "question": item.get("question", ""),
                    "conditions": normalize_media(raw_cond),
                    "options": {
                        "A": normalize_media(raw_opts.get("A", {})),
                        "B": normalize_media(raw_opts.get("B", {})),
                        "C": normalize_media(raw_opts.get("C", {})),
                        "D": normalize_media(raw_opts.get("D", {})),
                    },
                    "correct_answer": item.get("correct_answer", ""),
                    "category": item.get("category") or subtask,
                }
                combos[combo].append(row)

    # ── Step 4: sort by (subtask, index) and write JSONL ───────────────────
    os.makedirs(out_dir, exist_ok=True)
    for combo, rows in sorted(combos.items()):
        rows.sort(key=lambda r: (r["subtask"], r["index"]))

        out_path = os.path.join(out_dir, f"{combo}.jsonl")
        with open(out_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        counts: dict[str, int] = defaultdict(int)
        for r in rows:
            counts[r["subtask"]] += 1
        print(f"\n{combo}: {len(rows)} rows total")
        for st in sorted(counts):
            print(f"  {st}: {counts[st]}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-root", default=DEFAULT_TASKS_ROOT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", default=DEFAULT_SEED, type=int)
    args = parser.parse_args()

    build(args.tasks_root, args.out_dir, args.seed)
