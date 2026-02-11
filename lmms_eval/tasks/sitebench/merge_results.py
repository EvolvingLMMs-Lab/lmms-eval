#!/usr/bin/env python3
"""
Merge SiteBench Image and Video results from lmms-eval output.

This script combines site_bench_image and site_bench_video results to compute
overall metrics matching VLMEvalKit's methodology.

Usage:
    python -m lmms_eval.tasks.sitebench.merge_results --logs-dir logs/MODEL_NAME/
    python -m lmms_eval.tasks.sitebench.merge_results --image-jsonl path/to/image.jsonl --video-jsonl path/to/video.jsonl
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import pandas as pd


def _empty_stats():
    return {
        "caa_num": 0.0,
        "caa_den": 0.0,
        "acc_num": 0.0,
        "acc_den": 0.0,
    }


def _count_options_from_input(text: str) -> int | None:
    """Count number of options from the input text."""
    if not text:
        return None
    lines = [line.strip() for line in text.splitlines()]
    try:
        start_idx = next(i for i, line in enumerate(lines) if line.lower().startswith("options"))
    except StopIteration:
        start_idx = None
    if start_idx is None:
        return None
    count = 0
    for line in lines[start_idx + 1 :]:
        if not line:
            break
        lower = line.lower()
        if lower.startswith("give me") or "best answer" in lower:
            break
        if re.match(r"^[A-Z]:", line):
            count += 1
        else:
            if count > 0:
                break
    return count if count > 0 else None


def _count_options_from_doc(doc: dict) -> int | None:
    """Count number of options from the doc dict."""
    if not isinstance(doc, dict):
        return None
    for key in ("choices", "options", "answer_choices"):
        value = doc.get(key)
        if isinstance(value, list) and len(value) > 0:
            return len(value)
    return None


def compute_random_expected_acc(jsonl_path: str) -> tuple[float, int, int]:
    """
    Compute the random expected accuracy (1/num_options average).

    Returns:
        tuple of (avg_random_acc, total_counted, missing_count)
    """
    total = 0
    sum_expect = 0.0
    missing = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            n_opt = _count_options_from_input(item.get("input"))
            if n_opt is None:
                n_opt = _count_options_from_doc(item.get("doc"))
            if n_opt is None or n_opt <= 0:
                missing += 1
                continue
            sum_expect += 1.0 / n_opt
            total += 1

    avg = sum_expect / total if total > 0 else 0.0
    return avg, total, missing


def compute_stats_from_jsonl(jsonl_path: str) -> dict:
    """
    Compute aggregated statistics from a samples JSONL file.

    Returns:
        dict with keys: metric_stats, category_stats, overall
    """
    metric_stats = defaultdict(_empty_stats)
    category_stats = defaultdict(_empty_stats)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)

            # Get accuracy and chance_adjusted_acc dicts
            acc = item.get("accuracy", {})
            caa = item.get("chance_adjusted_acc", {})

            acc_total = acc.get("total", 0.0)
            caa_total = caa.get("total", 0.0)

            # Update metric stats (by category/dataset keys)
            for key, value in acc.items():
                if key == "total":
                    continue
                metric_stats[key]["acc_num"] += value
                metric_stats[key]["acc_den"] += acc_total

            for key, value in caa.items():
                if key == "total":
                    continue
                metric_stats[key]["caa_num"] += value
                metric_stats[key]["caa_den"] += caa_total

            # Extract category from doc if available
            doc = item.get("doc")
            if isinstance(doc, dict):
                category = doc.get("category")
                if category:
                    category_stats[category]["acc_num"] += acc.get("overall", 0.0)
                    category_stats[category]["acc_den"] += acc_total
                    category_stats[category]["caa_num"] += caa.get("overall", 0.0)
                    category_stats[category]["caa_den"] += caa_total

    # Compute overall from "overall" key in metric_stats
    overall = None
    if "overall" in metric_stats:
        overall = metric_stats["overall"]

    return {
        "metric_stats": dict(metric_stats),
        "category_stats": dict(category_stats),
        "overall": overall,
    }


def stats_to_df(stats: dict, label_col: str) -> pd.DataFrame:
    """Convert stats dict to a pandas DataFrame."""
    rows = []
    for key, val in stats.items():
        caa = val["caa_num"] / val["caa_den"] if val["caa_den"] > 0 else 0.0
        acc = val["acc_num"] / val["acc_den"] if val["acc_den"] > 0 else 0.0
        count = val["acc_den"] if val["acc_den"] > 0 else val["caa_den"]
        rows.append((key, caa * 100, acc * 100, int(count)))

    df = pd.DataFrame(rows, columns=[label_col, "CAA (%)", "Accuracy (%)", "Count"])
    df = df.sort_values(by="CAA (%)", ascending=False, ignore_index=True)
    return df


def merge_stats(stats1: dict, stats2: dict) -> dict:
    """Merge two stats dictionaries."""
    merged = defaultdict(_empty_stats)

    for key, val in stats1.items():
        merged[key]["acc_num"] += val["acc_num"]
        merged[key]["acc_den"] += val["acc_den"]
        merged[key]["caa_num"] += val["caa_num"]
        merged[key]["caa_den"] += val["caa_den"]

    for key, val in stats2.items():
        merged[key]["acc_num"] += val["acc_num"]
        merged[key]["acc_den"] += val["acc_den"]
        merged[key]["caa_num"] += val["caa_num"]
        merged[key]["caa_den"] += val["caa_den"]

    return dict(merged)


def find_latest_sitebench_files(logs_dir: str) -> tuple[str | None, str | None]:
    """
    Find the latest site_bench_image and site_bench_video JSONL files.

    Returns:
        tuple of (image_jsonl_path, video_jsonl_path)
    """
    # Find all site_bench_image JSONL files
    image_files = glob.glob(os.path.join(logs_dir, "*samples_site_bench_image.jsonl"))
    # Find all site_bench_video JSONL files (including 32frame_multiimage variants)
    video_files = glob.glob(os.path.join(logs_dir, "*samples_site_bench_video*.jsonl"))

    # Sort by filename (timestamp) descending to get latest
    image_files.sort(reverse=True)
    video_files.sort(reverse=True)

    image_path = image_files[0] if image_files else None
    video_path = video_files[0] if video_files else None

    return image_path, video_path


def print_results(name: str, stats: dict, category_stats: dict = None, random_acc: float = None):
    """Print formatted results."""
    print(f"\n{'='*60}")
    print(f"{name}")
    print("=" * 60)

    if stats.get("overall"):
        overall = stats["overall"]
        acc = overall["acc_num"] / overall["acc_den"] if overall["acc_den"] > 0 else 0.0
        caa = overall["caa_num"] / overall["caa_den"] if overall["caa_den"] > 0 else 0.0
        count = int(overall["acc_den"])
        print(f"Overall: Accuracy={acc*100:.2f}%, CAA={caa*100:.2f}%, Count={count}")
        if random_acc is not None:
            print(f"Random Expected Accuracy: {random_acc*100:.2f}%")

    if category_stats:
        cat_df = stats_to_df(category_stats, "Category")
        print("\nCategory Breakdown:")
        print(cat_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Merge SiteBench Image and Video results from lmms-eval output.")
    parser.add_argument(
        "--logs-dir",
        type=str,
        help="Path to the model's logs directory (e.g., logs/MODEL_NAME/). " "Will auto-detect the latest site_bench_image and site_bench_video files.",
    )
    parser.add_argument(
        "--image-jsonl",
        type=str,
        help="Path to site_bench_image samples JSONL file.",
    )
    parser.add_argument(
        "--video-jsonl",
        type=str,
        help="Path to site_bench_video samples JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional output JSON file to save combined results.",
    )

    args = parser.parse_args()

    image_stats = None
    video_stats = None
    image_path = None
    video_path = None
    image_random_acc = None
    video_random_acc = None

    # Auto-detect files from logs directory
    if args.logs_dir:
        image_path, video_path = find_latest_sitebench_files(args.logs_dir)

        if image_path:
            print(f"Found image JSONL: {image_path}")
            image_stats = compute_stats_from_jsonl(image_path)
            image_random_acc, _, _ = compute_random_expected_acc(image_path)
        else:
            print("Warning: No site_bench_image JSONL found")

        if video_path:
            print(f"Found video JSONL: {video_path}")
            video_stats = compute_stats_from_jsonl(video_path)
            video_random_acc, _, _ = compute_random_expected_acc(video_path)
        else:
            print("Warning: No site_bench_video JSONL found")

    # Use explicit file paths if provided (override auto-detected)
    if args.image_jsonl:
        image_path = args.image_jsonl
        print(f"Using image JSONL: {image_path}")
        image_stats = compute_stats_from_jsonl(image_path)
        image_random_acc, _, _ = compute_random_expected_acc(image_path)

    if args.video_jsonl:
        video_path = args.video_jsonl
        print(f"Using video JSONL: {video_path}")
        video_stats = compute_stats_from_jsonl(video_path)
        video_random_acc, _, _ = compute_random_expected_acc(video_path)

    # Print individual results
    if image_stats:
        print_results(
            "SiteBench Image",
            image_stats,
            image_stats.get("category_stats"),
            image_random_acc,
        )

    if video_stats:
        print_results(
            "SiteBench Video",
            video_stats,
            video_stats.get("category_stats"),
            video_random_acc,
        )

    # Compute and print combined results
    if image_stats and video_stats:
        combined_metric = merge_stats(
            image_stats.get("metric_stats", {}),
            video_stats.get("metric_stats", {}),
        )
        combined_category = merge_stats(
            image_stats.get("category_stats", {}),
            video_stats.get("category_stats", {}),
        )

        # Compute combined overall
        img_overall = image_stats.get("overall", _empty_stats())
        vid_overall = video_stats.get("overall", _empty_stats())
        combined_overall = {
            "acc_num": img_overall["acc_num"] + vid_overall["acc_num"],
            "acc_den": img_overall["acc_den"] + vid_overall["acc_den"],
            "caa_num": img_overall["caa_num"] + vid_overall["caa_num"],
            "caa_den": img_overall["caa_den"] + vid_overall["caa_den"],
        }

        combined_stats = {
            "metric_stats": combined_metric,
            "category_stats": combined_category,
            "overall": combined_overall,
        }

        # Compute combined random expected accuracy (weighted average)
        combined_random_acc = None
        if image_random_acc is not None and video_random_acc is not None:
            img_count = image_stats["overall"]["acc_den"]
            vid_count = video_stats["overall"]["acc_den"]
            total_count = img_count + vid_count
            if total_count > 0:
                combined_random_acc = (image_random_acc * img_count + video_random_acc * vid_count) / total_count

        print_results(
            "SiteBench Combined (Image + Video)",
            combined_stats,
            combined_category,
            combined_random_acc,
        )

        # Save to output file if requested
        if args.output:
            output_data = {
                "image": {
                    "file": image_path,
                    "accuracy": (image_stats["overall"]["acc_num"] / image_stats["overall"]["acc_den"] * 100 if image_stats["overall"]["acc_den"] > 0 else 0),
                    "caa": (image_stats["overall"]["caa_num"] / image_stats["overall"]["caa_den"] * 100 if image_stats["overall"]["caa_den"] > 0 else 0),
                    "count": int(image_stats["overall"]["acc_den"]),
                },
                "video": {
                    "file": video_path,
                    "accuracy": (video_stats["overall"]["acc_num"] / video_stats["overall"]["acc_den"] * 100 if video_stats["overall"]["acc_den"] > 0 else 0),
                    "caa": (video_stats["overall"]["caa_num"] / video_stats["overall"]["caa_den"] * 100 if video_stats["overall"]["caa_den"] > 0 else 0),
                    "count": int(video_stats["overall"]["acc_den"]),
                },
                "combined": {
                    "accuracy": (combined_overall["acc_num"] / combined_overall["acc_den"] * 100 if combined_overall["acc_den"] > 0 else 0),
                    "caa": (combined_overall["caa_num"] / combined_overall["caa_den"] * 100 if combined_overall["caa_den"] > 0 else 0),
                    "count": int(combined_overall["acc_den"]),
                },
            }
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to: {args.output}")

    elif not image_stats and not video_stats:
        print("\nError: No SiteBench results found!")
        print("Please provide:")
        print("  --logs-dir path/to/model/logs/")
        print("  OR --image-jsonl and --video-jsonl paths")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
