import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def _parse_args():
    parser = argparse.ArgumentParser(description="Benchmark MINERVA pipeline latency: local raw videos vs Lance storage")
    parser.add_argument("--local-video-dir", type=Path, required=True, help="Local raw MINERVA video directory")
    parser.add_argument("--lance-uri", type=str, required=True, help="Lance dataset URI for MINERVA videos")
    parser.add_argument("--lance-cache-dir", type=Path, default=None, help="Optional cache dir for Lance-resolved videos")
    parser.add_argument("--limit", type=int, default=100, help="Number of samples to run")
    parser.add_argument("--batch-size", type=int, default=1, help="lmms_eval batch size")
    parser.add_argument("--decode-num-frames", type=int, default=8, help="Decode frame count in the fixed model pipeline path")
    parser.add_argument("--decode-fps", type=float, default=None, help="Optional fps for fixed decode path")
    parser.add_argument("--read-bytes", type=int, default=65536, help="Bytes to read when decode-num-frames=0")
    parser.add_argument("--output-root", type=Path, default=Path("./logs/minerva_pipeline_latency"), help="Root directory for run artifacts")
    parser.add_argument("--verbosity", type=str, default="INFO", help="lmms_eval verbosity")
    parser.add_argument("--skip-local", action="store_true", help="Skip local mode run")
    parser.add_argument("--skip-lance", action="store_true", help="Skip lance mode run")
    return parser.parse_args()


def _build_model_args(metrics_path: Path, args) -> str:
    parts = [
        f"read_bytes={args.read_bytes}",
        "response=A",
        "allow_remote=false",
        "fail_on_missing=true",
        f"decode_num_frames={args.decode_num_frames}",
        f"metrics_output_path={metrics_path}",
    ]
    if args.decode_fps is not None:
        parts.append(f"decode_fps={args.decode_fps}")
    return ",".join(parts)


def _build_env(base_env: dict, mode: str, args) -> dict:
    env = dict(base_env)
    env.pop("MINERVA_VIDEO_DIR", None)
    env.pop("MINERVA_LANCE_VIDEO_URI", None)
    env.pop("MINERVA_LANCE_CACHE_DIR", None)
    env.pop("MINERVA_LANCE_VIDEO_ID_COLUMN", None)
    env.pop("MINERVA_LANCE_VIDEO_BLOB_COLUMN", None)

    if mode == "local":
        env["MINERVA_VIDEO_DIR"] = str(args.local_video_dir)
    else:
        env["MINERVA_LANCE_VIDEO_URI"] = args.lance_uri
        if args.lance_cache_dir is not None:
            env["MINERVA_LANCE_CACHE_DIR"] = str(args.lance_cache_dir)
    return env


def _run_mode(mode: str, args, base_env: dict) -> dict:
    mode_dir = args.output_root / mode
    eval_output_dir = mode_dir / "eval_output"
    metrics_path = mode_dir / "dummy_metrics.json"
    mode_dir.mkdir(parents=True, exist_ok=True)

    model_args = _build_model_args(metrics_path, args)
    command = [
        "uv",
        "run",
        "--with",
        "pylance",
        "--with",
        "pyarrow",
        "python",
        "-m",
        "lmms_eval",
        "--model",
        "dummy_video_reader",
        "--model_args",
        model_args,
        "--tasks",
        "minerva",
        "--batch_size",
        str(args.batch_size),
        "--limit",
        str(args.limit),
        "--output_path",
        str(eval_output_dir),
        "--verbosity",
        args.verbosity,
    ]
    env = _build_env(base_env, mode, args)
    subprocess.run(command, check=True, env=env)

    if not metrics_path.exists():
        raise FileNotFoundError(f"Dummy model metrics file not found: {metrics_path}")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _extract_metric(metrics: dict[str, Any], path: tuple[str, ...], default: Any = 0.0) -> Any:
    value = metrics
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _extract_float(metrics: dict[str, Any], path: tuple[str, ...], default: float = 0.0) -> float:
    value = _extract_metric(metrics, path, default)
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _extract_dict(metrics: dict[str, Any], path: tuple[str, ...]) -> dict[str, Any]:
    value = _extract_metric(metrics, path, {})
    if isinstance(value, dict):
        return value
    return {}


def _print_mode_summary(mode: str, metrics: dict[str, Any]):
    total = _extract_dict(metrics, ("latency", "total"))
    resolve_stats = _extract_dict(metrics, ("latency", "resolve"))
    decode_stats = _extract_dict(metrics, ("latency", "decode"))

    print(f"{mode}_samples={total.get('samples', 0)}")
    print(f"{mode}_total_mean_ms={total.get('mean_ms', 0.0):.3f}")
    print(f"{mode}_total_p50_ms={total.get('p50_ms', 0.0):.3f}")
    print(f"{mode}_total_p95_ms={total.get('p95_ms', 0.0):.3f}")
    print(f"{mode}_resolve_mean_ms={resolve_stats.get('mean_ms', 0.0):.3f}")
    print(f"{mode}_decode_mean_ms={decode_stats.get('mean_ms', 0.0):.3f}")
    print(f"{mode}_videos_per_s={metrics.get('throughput_videos_per_s', 0.0):.3f}")
    print(f"{mode}_decode_frames_per_s={metrics.get('throughput_decode_frames_per_s', 0.0):.3f}")


def _print_comparison(local_metrics: dict[str, Any], lance_metrics: dict[str, Any]):
    local_total_mean = _extract_float(local_metrics, ("latency", "total", "mean_ms"), 0.0)
    lance_total_mean = _extract_float(lance_metrics, ("latency", "total", "mean_ms"), 0.0)
    local_resolve_mean = _extract_float(local_metrics, ("latency", "resolve", "mean_ms"), 0.0)
    lance_resolve_mean = _extract_float(lance_metrics, ("latency", "resolve", "mean_ms"), 0.0)

    if local_total_mean > 0:
        print(f"ratio_total_mean_lance_over_local={lance_total_mean / local_total_mean:.3f}")
    else:
        print("ratio_total_mean_lance_over_local=0.000")

    if local_resolve_mean > 0:
        print(f"ratio_resolve_mean_lance_over_local={lance_resolve_mean / local_resolve_mean:.3f}")
    else:
        print("ratio_resolve_mean_lance_over_local=0.000")


def main():
    args = _parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    base_env = os.environ.copy()

    run_local = not args.skip_local
    run_lance = not args.skip_lance
    if not run_local and not run_lance:
        raise ValueError("At least one mode must run; remove --skip-local and/or --skip-lance")

    local_metrics = None
    lance_metrics = None

    if run_local:
        local_metrics = _run_mode("local", args, base_env)
        _print_mode_summary("local", local_metrics)

    if run_lance:
        lance_metrics = _run_mode("lance", args, base_env)
        _print_mode_summary("lance", lance_metrics)

    if local_metrics is not None and lance_metrics is not None:
        _print_comparison(local_metrics, lance_metrics)


if __name__ == "__main__":
    main()
