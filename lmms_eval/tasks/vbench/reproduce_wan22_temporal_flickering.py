"""Re-score the public Wan2.2 sample archive's temporal-flickering videos.

This is a lightweight reproduction of ``vbench/temporal_flickering.py`` at
the source revision used by this integration.  It deliberately validates the
complete 75-prompt x 5-sample subset before reporting a score; a partial
download must not silently turn into a benchmark result.

Run with uv, for example::

    uv run --no-project --with 'numpy<2' --with 'opencv-python-headless<5' python \
        lmms_eval/tasks/vbench/reproduce_wan22_temporal_flickering.py \
        --videos-dir /data/Wan2.2-T2V-A14B/videos \
        --output-json /data/results/wan22_temporal_flickering.json
"""

from __future__ import annotations

import argparse
import json
import platform
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import cv2
import numpy as np

SOURCE_REVISION = "1ee42dada7a2f7cfaf4290e8a02d087f6f8ee425"
FULL_INFO_URL = f"https://raw.githubusercontent.com/Vchitect/VBench/{SOURCE_REVISION}/vbench/VBench_full_info.json"
SAMPLE_ARCHIVE_URL = "https://drive.google.com/drive/folders/1Zb3YQr2YKXY2WqxcI_s0uSeup7Lw5klL"
LEADERBOARD_URL = "https://huggingface.co/spaces/Vchitect/VBench_Leaderboard"
OFFICIAL_PERCENT = 98.92
EXPECTED_PROMPTS = 75
SAMPLES_PER_PROMPT = 5


def load_registry(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        with urlopen(FULL_INFO_URL) as response:
            payload = response.read()
    else:
        payload = path.read_bytes()
    records = json.loads(payload)
    if not isinstance(records, list):
        raise ValueError("VBench full-info registry must contain a list")
    return records


def expected_video_paths(videos_dir: Path, records: list[dict[str, Any]]) -> list[Path]:
    prompts = [record["prompt_en"] for record in records if "temporal_flickering" in record.get("dimension", [])]
    if len(prompts) != EXPECTED_PROMPTS or len(set(prompts)) != EXPECTED_PROMPTS:
        raise ValueError(f"Expected {EXPECTED_PROMPTS} unique temporal-flickering prompts, found {len(set(prompts))}")
    return [videos_dir / f"{prompt}-{sample_index}.mp4" for prompt in prompts for sample_index in range(SAMPLES_PER_PROMPT)]


def score_video(path_string: str) -> dict[str, Any]:
    """Apply the exact adjacent-frame MAE formula used by VBench v0.1.5."""

    path = Path(path_string)
    frames = []
    video = cv2.VideoCapture(str(path))
    fps = float(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        frames.append(frame)
    video.release()
    if not frames:
        raise ValueError(f"No frames decoded from {path}")

    frame_mae = [np.mean(cv2.absdiff(np.array(first, dtype=np.float32), np.array(second, dtype=np.float32))) for first, second in zip(frames[:-1], frames[1:])]
    score = (255.0 - np.mean(np.array(frame_mae)).item()) / 255.0
    return {
        "video_path": str(path),
        "video_results": score,
        "frames": len(frames),
        "fps": fps,
        "width": width,
        "height": height,
    }


def reproduce(videos_dir: Path, full_info: Path | None, workers: int) -> dict[str, Any]:
    paths = expected_video_paths(videos_dir, load_registry(full_info))
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} of {len(paths)} expected videos; first: {missing[0]}")

    with ProcessPoolExecutor(max_workers=workers) as pool:
        details = list(pool.map(score_video, map(str, paths), chunksize=1))
    aggregate = float(np.mean([item["video_results"] for item in details]))
    metadata = sorted({(item["width"], item["height"], item["frames"], item["fps"]) for item in details})
    measured_percent = 100 * aggregate
    return {
        "provenance": {
            "vbench_revision": SOURCE_REVISION,
            "full_info_url": FULL_INFO_URL,
            "sample_archive_url": SAMPLE_ARCHIVE_URL,
            "leaderboard_url": LEADERBOARD_URL,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
        },
        "video_count": len(details),
        "video_metadata": [{"width": width, "height": height, "frames": frames, "fps": fps} for width, height, frames, fps in metadata],
        "official_percent": OFFICIAL_PERCENT,
        "measured_percent": measured_percent,
        "delta_percent": measured_percent - OFFICIAL_PERCENT,
        "temporal_flickering": [aggregate, details],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--full-info", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = reproduce(args.videos_dir, args.full_info, args.workers)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {key: value for key, value in result.items() if key != "temporal_flickering"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
