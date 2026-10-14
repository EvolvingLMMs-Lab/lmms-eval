"""Release file primitives, including streaming SHA-256 on Python 3.10."""

import hashlib
import json
from pathlib import PurePosixPath


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or str(relative) != name:
        raise ValueError(f"Unsafe path: {name}")
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError(f"Unsafe destination: {name}")
    return path


def selected_videos(data, video_ids):
    videos = rows(data / "videos.jsonl")
    available = {v["video_id"] for v in videos}
    if len(available) != len(videos):
        raise ValueError("Duplicate video IDs")
    if video_ids and not set(video_ids) <= available:
        raise ValueError("Unknown video IDs")
    return [v for v in videos if not video_ids or v["video_id"] in video_ids]
