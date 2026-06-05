"""Shared helper for Qwen-native point parsing.

Qwen-VL grounding models emit points as a JSON list of
``{"point_2d": [x, y], ...}`` entries on a 0-1000 normalized grid. The pointing
tasks (where2place / refspatial / pointbench) ship ``*_json`` variants that
prompt for this format and parse it with :func:`parse_point2d` below.
"""

import re

import numpy as np


def parse_point2d(text: str, width: int, height: int) -> np.ndarray:
    """Parse ``[x, y]`` / ``(x, y)`` coordinate pairs into pixel points.

    Integers are interpreted on a 0-1000 grid (divided by 1000, then scaled to
    the image); floats are treated as already-normalized in [0, 1]. This covers
    Qwen's native ``"point_2d": [x, y]`` JSON output as well as bare tuples.
    """
    pattern = r"[\[\(]\s*([-+]?\d+\.?\d*)\s*,\s*([-+]?\d+\.?\d*)\s*[\]\)]"
    points = []
    for xs, ys in re.findall(pattern, text):
        xf, yf = float(xs), float(ys)
        is_float = ("." in xs) or ("." in ys)
        if is_float:
            x, y = int(xf * width), int(yf * height)
        else:
            x, y = int(xf / 1000 * width), int(yf / 1000 * height)
        points.append((x, y))
    return np.array(points)
