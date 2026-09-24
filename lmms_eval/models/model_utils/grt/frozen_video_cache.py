"""Immutable post-resize frame caches for reproducible video evaluation.

The legacy Decord decoder is not deterministic across independent processes on
all of the benchmark videos.  A frozen cache therefore stores the exact RGB
arrays consumed by the image processor.  Both the builder and runtime use the
resize helper in this module, and the runtime fails closed on every provenance
or content mismatch.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import PIL

FROZEN_DECORD_SCHEMA_VERSION = 1
FROZEN_DECORD_BACKEND = "decord_frozen_post_resize_v1"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    """Hash the canonical C-contiguous array payload, excluding NPY headers."""

    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def resize_video_frames(frames: np.ndarray, max_image_size: Optional[int]) -> np.ndarray:
    """Apply the exact historical LLaVA-HF max-side resize to RGB frames."""

    array = np.asarray(frames)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"Expected RGB video array [T,H,W,3], got shape={array.shape}")
    if array.dtype != np.uint8:
        raise ValueError(f"Expected uint8 video frames, got dtype={array.dtype}")
    if array.shape[0] <= 0 or array.shape[1] <= 0 or array.shape[2] <= 0:
        raise ValueError(f"Video frames must be non-empty, got shape={array.shape}")
    if max_image_size in (None, 0):
        return np.ascontiguousarray(array)

    max_side = int(max_image_size)
    if max_side <= 0:
        raise ValueError(f"max_image_size must be positive, got {max_image_size}")
    resized_frames = []
    for frame in array:
        image = PIL.Image.fromarray(frame)
        width, height = image.size
        scale = min(float(max_side) / max(width, height), 1.0)
        if scale < 1.0:
            new_size = (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            )
            image = image.resize(new_size, PIL.Image.Resampling.BICUBIC)
        resized_frames.append(np.asarray(image, dtype=np.uint8))
    return np.ascontiguousarray(np.stack(resized_frames, axis=0))


@dataclass(frozen=True)
class FrozenVideoFrames:
    frames: np.ndarray
    video_key: str
    total_frames: int
    avg_fps: float
    frame_indices: tuple[int, ...]
    array_sha256: str


class FrozenDecordFrameCache:
    """Validated, immutable lookup over a frozen Decord frame manifest."""

    def __init__(
        self,
        manifest_path: str,
        *,
        requested_frames: int,
        max_image_size: Optional[int],
    ) -> None:
        if not manifest_path:
            raise ValueError("decord_frame_cache_manifest is required for decord_frozen")
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Frozen Decord manifest does not exist: {self.manifest_path}")
        self.root = self.manifest_path.parent.resolve()
        self.manifest_sha256 = sha256_file(self.manifest_path)
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid frozen Decord manifest: {self.manifest_path}") from exc
        if not isinstance(payload, dict):
            raise ValueError("Frozen Decord manifest root must be an object")
        self._validate_header(payload, requested_frames, max_image_size)

        entries = payload.get("entries")
        if not isinstance(entries, list) or not entries:
            raise ValueError("Frozen Decord manifest entries must be a non-empty list")
        expected_videos = payload.get("expected_videos")
        if not isinstance(expected_videos, int) or expected_videos <= 0:
            raise ValueError("Frozen Decord expected_videos must be a positive integer")
        if len(entries) != expected_videos:
            raise ValueError(f"Frozen Decord entry count mismatch: expected_videos={expected_videos}, entries={len(entries)}")

        self.requested_frames = int(requested_frames)
        self.max_image_size = None if max_image_size is None else int(max_image_size)
        self.dataset_revision = str(payload.get("dataset_revision") or "")
        if not self.dataset_revision:
            raise ValueError("Frozen Decord dataset_revision must be non-empty")
        self._entries: Dict[str, Mapping[str, Any]] = {}
        for raw_entry in entries:
            entry = self._validate_entry(raw_entry)
            key = str(entry["video_key"])
            basename = str(entry["video_basename"])
            for lookup_key in {key, basename}:
                existing = self._entries.get(lookup_key)
                if existing is not None and existing is not entry:
                    raise ValueError(f"Ambiguous frozen Decord lookup key in manifest: {lookup_key}")
                self._entries[lookup_key] = entry

    @staticmethod
    def _validate_header(
        payload: Mapping[str, Any],
        requested_frames: int,
        max_image_size: Optional[int],
    ) -> None:
        if payload.get("schema_version") != FROZEN_DECORD_SCHEMA_VERSION:
            raise ValueError(f"Unsupported frozen Decord schema_version={payload.get('schema_version')}; expected {FROZEN_DECORD_SCHEMA_VERSION}")
        if payload.get("backend") != FROZEN_DECORD_BACKEND:
            raise ValueError(f"Unsupported frozen Decord backend={payload.get('backend')}; expected {FROZEN_DECORD_BACKEND}")
        expected_frames = int(requested_frames)
        if payload.get("requested_frames") != expected_frames:
            raise ValueError(f"Frozen Decord requested_frames mismatch: manifest={payload.get('requested_frames')}, runtime={expected_frames}")
        runtime_max = None if max_image_size is None else int(max_image_size)
        if payload.get("max_image_size") != runtime_max:
            raise ValueError(f"Frozen Decord max_image_size mismatch: manifest={payload.get('max_image_size')}, runtime={runtime_max}")

    def _validate_entry(self, raw_entry: Any) -> Mapping[str, Any]:
        if not isinstance(raw_entry, dict):
            raise ValueError("Frozen Decord entries must be objects")
        required = {
            "video_key",
            "video_basename",
            "cache_file",
            "array_sha256",
            "shape",
            "dtype",
            "total_frames",
            "avg_fps",
            "frame_indices",
        }
        missing = sorted(required.difference(raw_entry))
        if missing:
            raise ValueError(f"Frozen Decord entry missing fields: {missing}")
        if not str(raw_entry["video_key"]).strip() or not str(raw_entry["video_basename"]).strip():
            raise ValueError("Frozen Decord video keys must be non-empty")
        if Path(str(raw_entry["video_basename"])).name != str(raw_entry["video_basename"]):
            raise ValueError("Frozen Decord video_basename must not contain directories")
        cache_path = self._contained_cache_path(str(raw_entry["cache_file"]))
        if not cache_path.is_file():
            raise FileNotFoundError(f"Frozen Decord frame array does not exist: {cache_path}")
        shape = raw_entry["shape"]
        if not isinstance(shape, list) or len(shape) != 4 or any(not isinstance(value, int) or value <= 0 for value in shape) or shape[0] != self.requested_frames or shape[-1] != 3:
            raise ValueError(f"Invalid frozen Decord shape for {raw_entry['video_key']}: {shape}")
        if raw_entry["dtype"] != "uint8":
            raise ValueError(f"Invalid frozen Decord dtype for {raw_entry['video_key']}: {raw_entry['dtype']}")
        indices = raw_entry["frame_indices"]
        if not isinstance(indices, list) or len(indices) != self.requested_frames or any(not isinstance(value, int) or value < 0 for value in indices):
            raise ValueError(f"Invalid frozen Decord frame_indices for {raw_entry['video_key']}")
        total_frames = raw_entry["total_frames"]
        if not isinstance(total_frames, int) or total_frames <= 0:
            raise ValueError("Frozen Decord total_frames must be a positive integer")
        if any(index >= total_frames for index in indices):
            raise ValueError("Frozen Decord frame index exceeds total_frames")
        if not isinstance(raw_entry["avg_fps"], (int, float)) or raw_entry["avg_fps"] < 0:
            raise ValueError("Frozen Decord avg_fps must be non-negative")
        array_sha = str(raw_entry["array_sha256"])
        if not re_full_sha256(array_sha):
            raise ValueError("Frozen Decord array_sha256 must be a lowercase SHA-256")
        file_sha = raw_entry.get("file_sha256")
        if file_sha is not None and not re_full_sha256(str(file_sha)):
            raise ValueError("Frozen Decord file_sha256 must be a lowercase SHA-256")
        return raw_entry

    def _contained_cache_path(self, relative_path: str) -> Path:
        candidate = Path(relative_path)
        if candidate.is_absolute():
            raise ValueError("Frozen Decord cache_file must be relative to the manifest")
        resolved = (self.root / candidate).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"Frozen Decord cache_file escapes manifest directory: {relative_path}") from exc
        return resolved

    def load(self, video_path: str) -> FrozenVideoFrames:
        raw = str(video_path)
        lookup_candidates = (raw, Path(raw).name)
        entry = next((self._entries[key] for key in lookup_candidates if key in self._entries), None)
        if entry is None:
            raise KeyError(f"Video is absent from frozen Decord manifest: {Path(raw).name}")
        cache_path = self._contained_cache_path(str(entry["cache_file"]))
        expected_file_sha = entry.get("file_sha256")
        if expected_file_sha is not None:
            actual_file_sha = sha256_file(cache_path)
            if actual_file_sha != expected_file_sha:
                raise ValueError(f"Frozen Decord file SHA mismatch for {entry['video_key']}: expected={expected_file_sha}, actual={actual_file_sha}")
        try:
            array = np.load(cache_path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Could not load frozen Decord array: {cache_path}") from exc
        expected_shape = tuple(entry["shape"])
        if array.shape != expected_shape or array.dtype != np.uint8:
            raise ValueError(f"Frozen Decord array contract mismatch for {entry['video_key']}: shape={array.shape}, dtype={array.dtype}")
        array = np.ascontiguousarray(array)
        actual_array_sha = sha256_array(array)
        if actual_array_sha != entry["array_sha256"]:
            raise ValueError(f"Frozen Decord array SHA mismatch for {entry['video_key']}: expected={entry['array_sha256']}, actual={actual_array_sha}")
        return FrozenVideoFrames(
            frames=array.copy(),
            video_key=str(entry["video_key"]),
            total_frames=int(entry["total_frames"]),
            avg_fps=float(entry["avg_fps"]),
            frame_indices=tuple(int(value) for value in entry["frame_indices"]),
            array_sha256=actual_array_sha,
        )


def re_full_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)
