import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from lmms_eval.models.model_utils.load_video import read_video


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark lmms-eval video decode backends")
    parser.add_argument("--video", type=Path, required=True, help="Path to a local video file")
    parser.add_argument("--backends", type=str, default="pyav,torchcodec,dali", help="Comma-separated backend list")
    parser.add_argument("--num-frames", type=int, default=8, help="Frames to sample")
    parser.add_argument("--fps", type=float, default=None, help="Optional fps limit")
    parser.add_argument("--iterations", type=int, default=30, help="Timed iterations")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--force-include-last-frame", action="store_true", help="Pass force_include_last_frame to decoder")
    parser.add_argument("--output", type=Path, default=None, help="Optional output JSON path")
    return parser.parse_args()


def _frame_hash(frames: np.ndarray) -> str:
    if frames.size == 0:
        return ""
    first = np.ascontiguousarray(frames[0])
    return hashlib.sha256(first.tobytes()).hexdigest()


def _single_run(backend: str, video_path: str, args: argparse.Namespace) -> tuple[float, np.ndarray]:
    started = time.perf_counter()
    frames = read_video(
        video_path,
        num_frm=args.num_frames,
        fps=args.fps,
        force_include_last_frame=args.force_include_last_frame,
        backend=backend,
    )
    elapsed = time.perf_counter() - started
    return elapsed, frames


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "std_ms": 0.0}

    sorted_vals = sorted(values)
    p50_idx = max(0, min(len(sorted_vals) - 1, int(0.50 * (len(sorted_vals) - 1))))
    p95_idx = max(0, min(len(sorted_vals) - 1, int(0.95 * (len(sorted_vals) - 1))))
    return {
        "mean_ms": statistics.mean(values) * 1000.0,
        "p50_ms": sorted_vals[p50_idx] * 1000.0,
        "p95_ms": sorted_vals[p95_idx] * 1000.0,
        "std_ms": (statistics.pstdev(values) if len(values) > 1 else 0.0) * 1000.0,
    }


def _run_backend(backend: str, video_path: str, args: argparse.Namespace) -> dict[str, Any]:
    for _ in range(max(0, args.warmup)):
        _single_run(backend, video_path, args)

    timings: list[float] = []
    last_frames: np.ndarray | None = None
    for _ in range(max(1, args.iterations)):
        elapsed, frames = _single_run(backend, video_path, args)
        timings.append(elapsed)
        last_frames = frames

    if last_frames is None:
        raise RuntimeError("No decoded frames captured")

    frame_count = int(last_frames.shape[0]) if last_frames.ndim >= 1 else 0
    pixels = int(np.prod(last_frames.shape[1:])) if last_frames.ndim >= 2 else 0
    avg_s = statistics.mean(timings)
    return {
        "status": "ok",
        "backend": backend,
        "shape": list(last_frames.shape),
        "dtype": str(last_frames.dtype),
        "frame_hash": _frame_hash(last_frames),
        "timing": _stats(timings),
        "throughput": {
            "videos_per_s": (1.0 / avg_s) if avg_s > 0 else 0.0,
            "frames_per_s": (frame_count / avg_s) if avg_s > 0 else 0.0,
            "pixels_per_s": (pixels / avg_s) if avg_s > 0 else 0.0,
        },
    }


def _run_backend_safe(backend: str, video_path: str, args: argparse.Namespace) -> dict[str, Any]:
    try:
        return _run_backend(backend, video_path, args)
    except Exception as exc:
        return {
            "status": "error",
            "backend": backend,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _compute_relatives(results: list[dict[str, Any]]) -> None:
    baseline = None
    for item in results:
        if item.get("status") == "ok" and item.get("backend") == "pyav":
            baseline = item
            break
    if baseline is None:
        return

    base_mean_ms = baseline["timing"]["mean_ms"]
    for item in results:
        if item.get("status") != "ok":
            continue
        mean_ms = item["timing"]["mean_ms"]
        if base_mean_ms > 0 and mean_ms > 0:
            item["vs_pyav"] = {
                "latency_change_pct": ((mean_ms - base_mean_ms) / base_mean_ms) * 100.0,
                "speedup_x": base_mean_ms / mean_ms,
            }


def main() -> None:
    args = _parse_args()
    video_path = str(args.video.resolve())
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    backends = [b.strip().lower() for b in args.backends.split(",") if b.strip()]
    if not backends:
        raise ValueError("No backend selected")

    results = [_run_backend_safe(backend, video_path, args) for backend in backends]
    _compute_relatives(results)

    report: dict[str, Any] = {
        "video": video_path,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "force_include_last_frame": args.force_include_last_frame,
        "results": results,
    }

    rendered = json.dumps(report, indent=2, ensure_ascii=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
