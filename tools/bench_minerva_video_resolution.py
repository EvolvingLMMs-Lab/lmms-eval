import argparse
import json
import os
import random
import statistics
import time
from pathlib import Path

from lmms_eval.tasks.minerva import utils as minerva_utils


def _parse_args():
    parser = argparse.ArgumentParser(description="Benchmark MINERVA video resolution latency for local and Lance modes")
    parser.add_argument("--metadata-json", type=Path, required=True, help="Path to minerva.json")
    parser.add_argument("--mode", choices=["local", "lance"], required=True, help="Resolution mode to benchmark")
    parser.add_argument("--limit", type=int, default=100, help="Maximum docs to benchmark")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for doc sampling")
    parser.add_argument("--local-video-dir", type=Path, default=None, help="Required for local mode")
    parser.add_argument("--lance-uri", type=str, default=None, help="Required for lance mode")
    parser.add_argument("--lance-cache-dir", type=Path, default=None, help="Optional Lance cache dir")
    parser.add_argument("--sample-unique-video", action="store_true", help="Sample unique video_id rows before limiting")
    parser.add_argument("--allow-fallback", action="store_true", help="Allow fallback paths (not recommended for perf claims)")
    return parser.parse_args()


def _load_docs(metadata_json: Path):
    rows = json.loads(metadata_json.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("metadata-json must be a list")
    return rows


def _sample_docs(docs, limit: int, seed: int, sample_unique_video: bool):
    docs = [d for d in docs if str(d.get("video_id", "")).strip()]
    if sample_unique_video:
        uniq = {}
        for doc in docs:
            video_id = str(doc.get("video_id", "")).strip()
            if video_id not in uniq:
                uniq[video_id] = doc
        docs = list(uniq.values())

    if limit <= 0 or limit >= len(docs):
        return docs
    rng = random.Random(seed)
    return rng.sample(docs, k=limit)


def _configure_env(args):
    os.environ.pop("MINERVA_VIDEO_DIR", None)
    os.environ.pop("MINERVA_LANCE_VIDEO_URI", None)
    os.environ.pop("MINERVA_LANCE_CACHE_DIR", None)
    os.environ.pop("MINERVA_LANCE_VIDEO_ID_COLUMN", None)
    os.environ.pop("MINERVA_LANCE_VIDEO_BLOB_COLUMN", None)

    if args.mode == "local":
        if args.local_video_dir is None:
            raise ValueError("--local-video-dir is required for --mode local")
        os.environ["MINERVA_VIDEO_DIR"] = str(args.local_video_dir)
    else:
        if not args.lance_uri:
            raise ValueError("--lance-uri is required for --mode lance")
        os.environ["MINERVA_LANCE_VIDEO_URI"] = args.lance_uri
        if args.lance_cache_dir is not None:
            os.environ["MINERVA_LANCE_CACHE_DIR"] = str(args.lance_cache_dir)


def _is_under_dir(path: str, parent: Path) -> bool:
    try:
        Path(path).resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _assert_resolution_path(resolved: str, args):
    if args.allow_fallback:
        return

    if args.mode == "local":
        if args.local_video_dir is None:
            raise ValueError("--local-video-dir is required for strict local assertion")
        if not _is_under_dir(resolved, args.local_video_dir):
            raise RuntimeError(f"Unexpected non-local resolution in local mode: {resolved}")
    else:
        if resolved.startswith("https://www.youtube.com/watch?v="):
            raise RuntimeError(f"Unexpected YouTube fallback in lance mode: {resolved}")


def _run_once(docs, args):
    latencies = []
    for doc in docs:
        start = time.perf_counter()
        resolved = minerva_utils.minerva_doc_to_visual(doc)[0]
        _assert_resolution_path(str(resolved), args)
        latencies.append(time.perf_counter() - start)
    return latencies


def _print_stats(label: str, latencies):
    if not latencies:
        print(f"{label}: no samples")
        return

    sorted_vals = sorted(latencies)
    p50_idx = max(0, min(len(sorted_vals) - 1, int(0.50 * (len(sorted_vals) - 1))))
    p95_idx = max(0, min(len(sorted_vals) - 1, int(0.95 * (len(sorted_vals) - 1))))

    total = sum(latencies)
    print(f"{label} samples={len(latencies)}")
    print(f"{label} total_s={total:.6f}")
    print(f"{label} mean_ms={statistics.mean(latencies) * 1000:.3f}")
    print(f"{label} p50_ms={sorted_vals[p50_idx] * 1000:.3f}")
    print(f"{label} p95_ms={sorted_vals[p95_idx] * 1000:.3f}")
    print(f"{label} max_ms={max(latencies) * 1000:.3f}")


def main():
    args = _parse_args()
    _configure_env(args)
    minerva_utils._LANCE_RESOLVER = None

    docs = _sample_docs(_load_docs(args.metadata_json), limit=args.limit, seed=args.seed, sample_unique_video=args.sample_unique_video)

    startup_ms = None
    if args.mode == "lance":
        startup_begin = time.perf_counter()
        _ = minerva_utils._get_lance_resolver()
        startup_ms = (time.perf_counter() - startup_begin) * 1000

    cold_latencies = _run_once(docs, args)
    warm_latencies = _run_once(docs, args)

    if startup_ms is not None:
        print(f"startup_ms={startup_ms:.3f}")
    _print_stats("cold", cold_latencies)
    _print_stats("warm", warm_latencies)


if __name__ == "__main__":
    main()
