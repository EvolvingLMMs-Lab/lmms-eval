"""Pre-extract COVER videos and optionally dump the QA JSON.

This is a convenience script for manual setup.  During normal lmms-eval
runs, the task's utils.py handles everything automatically (downloads
the HF repo, extracts videos, loads JSONL data in process_docs).

Usage:
    # Extract videos only (recommended before first eval run):
    python -m lmms_eval.tasks.cover.generate_qa --extract-videos

    # Also dump a standalone QA JSON for inspection:
    python -m lmms_eval.tasks.cover.generate_qa \
        --extract-videos --output $HF_HOME/cover/cover_qa.json
"""

import argparse
import io
import json
import os
import zipfile
from collections import defaultdict

from huggingface_hub import snapshot_download

DATASET_REPO_ID = "PeterPanonly/COVER"


def main():
    parser = argparse.ArgumentParser(description="COVER dataset setup")
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output JSON file path (optional). If set, writes a flat " "QA JSON with all samples.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="",
        help="Directory for extracted videos. Default: $HF_HOME/cover/",
    )
    parser.add_argument(
        "--extract-videos",
        action="store_true",
        help="Extract VIDEO.zip into the cache directory.",
    )
    args = parser.parse_args()

    hf_home = os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface"))
    cache_dir = args.cache_dir or os.path.join(hf_home, "cover")

    # Download the HF repo
    print(f"Downloading {DATASET_REPO_ID} ...")
    repo_dir = snapshot_download(
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        etag_timeout=60,
    )
    print(f"  Repo cached at: {repo_dir}")

    # Extract videos
    if args.extract_videos:
        video_dir = os.path.join(cache_dir, "VIDEO")
        if os.path.exists(video_dir):
            print(f"  VIDEO directory already exists: {video_dir}")
        else:
            video_zip = os.path.join(repo_dir, "VIDEO.zip")
            if not os.path.exists(video_zip):
                print(f"  ERROR: VIDEO.zip not found at {video_zip}")
                return
            print(f"  Extracting VIDEO.zip to {cache_dir} ...")
            os.makedirs(cache_dir, exist_ok=True)
            with zipfile.ZipFile(video_zip) as zf:
                zf.extractall(cache_dir)
            print("  Done.")

    # Optionally dump QA JSON
    if args.output:
        jsonl_zip_path = os.path.join(repo_dir, "jsonl.zip")
        if not os.path.exists(jsonl_zip_path):
            print(f"  ERROR: jsonl.zip not found at {jsonl_zip_path}")
            return

        samples = []
        idx = 0
        with zipfile.ZipFile(jsonl_zip_path) as zf:
            for name in sorted(zf.namelist()):
                if not name.endswith(".jsonl"):
                    continue
                aspect = os.path.basename(name).replace(".jsonl", "")
                with zf.open(name) as f:
                    for line in f:
                        entry = json.loads(line)
                        src = entry["src_dataset"]
                        vname = entry["video_name"]
                        text = entry["text"]

                        video_path = f"VIDEO/{src}/{vname}"

                        orig = text["original_qa"]
                        samples.append(
                            {
                                "idx": idx,
                                "video_path": video_path,
                                "src_dataset": src,
                                "video_name": vname,
                                "question": orig["qs"],
                                "choices": orig["choice"],
                                "answer": orig["ans"],
                                "qa_type": "original",
                                "aspect": aspect,
                            }
                        )
                        idx += 1

                        cf = text["counterfactual_qa"]
                        samples.append(
                            {
                                "idx": idx,
                                "video_path": video_path,
                                "src_dataset": src,
                                "video_name": vname,
                                "question": cf["qs"],
                                "choices": cf["choice"],
                                "answer": cf["ans"],
                                "qa_type": "counterfactual",
                                "aspect": aspect,
                            }
                        )
                        idx += 1

        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(samples, f, indent=2)

        print(f"\n  {len(samples)} QA samples written to {args.output}")

        by_type = defaultdict(int)
        by_aspect = defaultdict(int)
        for s in samples:
            by_type[s["qa_type"]] += 1
            by_aspect[s["aspect"]] += 1

        print(f"  By qa_type: {dict(by_type)}")
        print(f"  By aspect ({len(by_aspect)} categories):")
        for aspect in sorted(by_aspect):
            print(f"    {aspect}: {by_aspect[aspect]}")
    elif not args.extract_videos:
        print("Nothing to do. Use --extract-videos and/or --output.")


if __name__ == "__main__":
    main()
