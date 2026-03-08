#!/usr/bin/env python3
"""Watch batch heartbeats and cancel hung lmms-eval jobs."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def load_heartbeat(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(data, dict):
        return None

    data = dict(data)
    data.setdefault("rank", path.stem)
    data["_path"] = str(path)
    return data


def collect_heartbeats(watch_dir: Path) -> list[dict[str, Any]]:
    heartbeats = []
    for path in sorted(watch_dir.glob("rank_*.json")):
        heartbeat = load_heartbeat(path)
        if heartbeat is not None:
            heartbeats.append(heartbeat)
    return heartbeats


def find_stale_heartbeats(
    heartbeats: list[dict[str, Any]],
    *,
    now: float,
    timeout_seconds: float,
    stale_phases: set[str],
) -> list[dict[str, Any]]:
    stale = []
    for heartbeat in heartbeats:
        phase = str(heartbeat.get("phase", ""))
        updated_at = heartbeat.get("updated_at")
        if phase not in stale_phases or not isinstance(updated_at, (int, float)):
            continue

        age_seconds = now - float(updated_at)
        if age_seconds > timeout_seconds:
            entry = dict(heartbeat)
            entry["age_seconds"] = age_seconds
            stale.append(entry)
    return stale


def write_timeout_snapshot(
    snapshot_path: Path,
    *,
    detected_at: float,
    timeout_seconds: float,
    stale_heartbeats: list[dict[str, Any]],
    all_heartbeats: list[dict[str, Any]],
) -> None:
    payload = {
        "detected_at": detected_at,
        "timeout_seconds": timeout_seconds,
        "stale_heartbeats": stale_heartbeats,
        "all_heartbeats": all_heartbeats,
    }
    snapshot_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")


def run_timeout_action(action: str, job_id: str) -> int:
    if action == "exit":
        return 0
    if action == "scancel":
        if not job_id:
            raise ValueError("--job-id is required when --timeout-action=scancel")
        return subprocess.run(["scancel", job_id], check=False).returncode
    raise ValueError(f"Unsupported timeout action: {action}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch lmms-eval batch heartbeats and fail fast on hangs.")
    parser.add_argument("--watch-dir", required=True, help="Directory containing per-rank heartbeat JSON files.")
    parser.add_argument("--timeout-seconds", type=float, required=True, help="Maximum allowed age for a stale phase.")
    parser.add_argument("--poll-seconds", type=float, default=15.0, help="Polling interval while waiting.")
    parser.add_argument(
        "--stale-phases",
        default="encode_start,chat_start",
        help="Comma-separated heartbeat phases that should be treated as timed sections.",
    )
    parser.add_argument(
        "--timeout-action",
        choices=("exit", "scancel"),
        default="scancel",
        help="Action to run after a timeout is detected.",
    )
    parser.add_argument("--job-id", default="", help="Slurm job id used with --timeout-action=scancel.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    watch_dir = Path(args.watch_dir)
    watch_dir.mkdir(parents=True, exist_ok=True)
    stale_phases = {phase.strip() for phase in args.stale_phases.split(",") if phase.strip()}

    print(
        f"[watchdog] monitoring {watch_dir} with timeout={args.timeout_seconds}s "
        f"poll={args.poll_seconds}s phases={sorted(stale_phases)}",
        flush=True,
    )

    while True:
        now = time.time()
        heartbeats = collect_heartbeats(watch_dir)
        stale = find_stale_heartbeats(
            heartbeats,
            now=now,
            timeout_seconds=args.timeout_seconds,
            stale_phases=stale_phases,
        )
        if stale:
            snapshot_path = watch_dir / "timeout_snapshot.json"
            write_timeout_snapshot(
                snapshot_path,
                detected_at=now,
                timeout_seconds=args.timeout_seconds,
                stale_heartbeats=stale,
                all_heartbeats=heartbeats,
            )
            print(f"[watchdog] timeout detected; wrote snapshot to {snapshot_path}", flush=True)
            for heartbeat in stale:
                print(
                    "[watchdog] stale rank={rank} phase={phase} batch={batch} age={age:.1f}s tasks={tasks} docs={docs}".format(
                        rank=heartbeat.get("rank"),
                        phase=heartbeat.get("phase"),
                        batch=heartbeat.get("batch_idx"),
                        age=heartbeat.get("age_seconds", -1.0),
                        tasks=heartbeat.get("task_names"),
                        docs=heartbeat.get("doc_ids"),
                    ),
                    flush=True,
                )
            rc = run_timeout_action(args.timeout_action, str(args.job_id))
            if rc != 0:
                print(f"[watchdog] timeout action {args.timeout_action} exited with rc={rc}", flush=True)
            return 1

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
