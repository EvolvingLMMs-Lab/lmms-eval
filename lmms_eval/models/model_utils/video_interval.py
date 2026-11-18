"""Frame index span for a requested video interval."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


def resolve_video_bounds(url, start_time=None, end_time=None):
    """Return a path plus an interval, unwrapping a legacy metadata dict."""
    if isinstance(url, dict):
        if start_time is None:
            start_time = url.get("video_start", url.get("start_time"))
        if end_time is None:
            end_time = url.get("video_end", url.get("end_time"))
        url = url.get("path") or url.get("url") or url.get("video")
    if isinstance(url, (list, tuple)):
        if not url or not isinstance(url[0], str):
            raise TypeError(f"Unsupported video path: {url!r}")
        url = url[0]
    if not isinstance(url, str):
        raise TypeError(f"Unsupported video path: {url!r}")
    return url, start_time, end_time


def interval_index_span(total_frames: int, fps: float, start_time: Optional[float] = None, end_time: Optional[float] = None) -> tuple[int, int]:
    """Inclusive frame indexes for ``[start_time, end_time]`` seconds."""
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if start_time is None and end_time is None:
        return 0, total_frames - 1
    start = 0 if start_time is None else math.ceil(float(start_time) * fps)
    end = total_frames - 1 if end_time is None else math.floor(float(end_time) * fps)
    start = min(max(start, 0), total_frames - 1)
    end = min(max(end, 0), total_frames - 1)
    if start > end:
        raise ValueError(f"Empty video interval: frames {start}..{end} from {start_time}s to {end_time}s")
    return start, end


def uniform_indices(total_frames: int, fps: float, count: int, start_time: Optional[float] = None, end_time: Optional[float] = None) -> np.ndarray:
    """Evenly sample ``count`` indexes inside the requested interval."""
    if count <= 0:
        raise ValueError("count must be positive")
    start, end = interval_index_span(total_frames, fps, start_time, end_time)
    return np.linspace(start, end, count, dtype=int)
