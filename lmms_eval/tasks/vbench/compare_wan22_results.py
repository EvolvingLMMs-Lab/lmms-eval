"""Aggregate a VBench result JSON and compare it with official Wan2.2 scores.

The baselines are the percentages displayed by the official VBench leaderboard
for its two VBench-Team-evaluated Wan2.2 submissions on 2026-08-27:
https://huggingface.co/spaces/Vchitect/VBench_Leaderboard
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

QUALITY_DIMENSIONS = (
    "subject consistency",
    "background consistency",
    "temporal flickering",
    "motion smoothness",
    "dynamic degree",
    "aesthetic quality",
    "imaging quality",
)
SEMANTIC_DIMENSIONS = (
    "object class",
    "multiple objects",
    "human action",
    "color",
    "spatial relationship",
    "scene",
    "appearance style",
    "temporal style",
    "overall consistency",
)
NORMALIZATION = {
    "subject consistency": (0.1462, 1.0),
    "background consistency": (0.2615, 1.0),
    "temporal flickering": (0.6293, 1.0),
    "motion smoothness": (0.7060, 0.9975),
    "dynamic degree": (0.0, 1.0),
    "aesthetic quality": (0.0, 1.0),
    "imaging quality": (0.0, 1.0),
    "object class": (0.0, 1.0),
    "multiple objects": (0.0, 1.0),
    "human action": (0.0, 1.0),
    "color": (0.0, 1.0),
    "spatial relationship": (0.0, 1.0),
    "scene": (0.0, 0.8222),
    "appearance style": (0.0009, 0.2855),
    "temporal style": (0.0, 0.3640),
    "overall consistency": (0.0, 0.3640),
}
DIMENSION_WEIGHTS = {dimension: 1.0 for dimension in QUALITY_DIMENSIONS + SEMANTIC_DIMENSIONS}
DIMENSION_WEIGHTS["dynamic degree"] = 0.5

OFFICIAL = {
    "no_prompt": {
        "total": 82.61,
        "quality": 85.03,
        "semantic": 72.92,
        "dimensions": {
            "subject consistency": 97.46,
            "background consistency": 96.67,
            "temporal flickering": 98.92,
            "motion smoothness": 97.93,
            "dynamic degree": 68.24,
            "aesthetic quality": 63.52,
            "imaging quality": 71.69,
            "object class": 85.04,
            "multiple objects": 74.61,
            "human action": 83.40,
            "color": 88.93,
            "spatial relationship": 80.22,
            "scene": 34.38,
            "appearance style": 20.27,
            "temporal style": 23.13,
            "overall consistency": 24.68,
        },
    },
    "qwen": {
        "total": 84.23,
        "quality": 85.42,
        "semantic": 79.50,
        "dimensions": {
            "subject consistency": 97.29,
            "background consistency": 97.39,
            "temporal flickering": 99.22,
            "motion smoothness": 98.16,
            "dynamic degree": 61.02,
            "aesthetic quality": 67.22,
            "imaging quality": 71.75,
            "object class": 94.06,
            "multiple objects": 82.10,
            "human action": 96.40,
            "color": 87.43,
            "spatial relationship": 78.39,
            "scene": 56.80,
            "appearance style": 20.39,
            "temporal style": 23.64,
            "overall consistency": 26.12,
        },
    },
}


def canonical_dimension(name: str) -> str:
    return name.strip().lower().replace("_", " ")


def load_raw_scores(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    scores: dict[str, float] = {}
    for raw_name, value in payload.items():
        name = canonical_dimension(raw_name)
        if name not in NORMALIZATION:
            continue
        if isinstance(value, list):
            value = value[0]
        score = float(value)
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"Expected raw VBench score in [0, 1] for {name!r}, got {score}")
        scores[name] = score
    missing = set(NORMALIZATION) - set(scores)
    if missing:
        raise ValueError(f"Missing VBench dimensions: {sorted(missing)}")
    return scores


def aggregate(raw_scores: dict[str, float]) -> dict[str, float]:
    normalized = {dimension: (raw_scores[dimension] - minimum) / (maximum - minimum) * DIMENSION_WEIGHTS[dimension] for dimension, (minimum, maximum) in NORMALIZATION.items()}
    quality = sum(normalized[dimension] for dimension in QUALITY_DIMENSIONS) / sum(DIMENSION_WEIGHTS[dimension] for dimension in QUALITY_DIMENSIONS)
    semantic = sum(normalized[dimension] for dimension in SEMANTIC_DIMENSIONS) / sum(DIMENSION_WEIGHTS[dimension] for dimension in SEMANTIC_DIMENSIONS)
    total = (4 * quality + semantic) / 5
    return {"total": 100 * total, "quality": 100 * quality, "semantic": 100 * semantic}


def comparison(raw_scores: dict[str, float], baseline_name: str) -> dict:
    measured = aggregate(raw_scores)
    baseline = OFFICIAL[baseline_name]
    dimensions = {
        dimension: {
            "official": baseline["dimensions"][dimension],
            "measured": 100 * raw_scores[dimension],
            "delta": 100 * raw_scores[dimension] - baseline["dimensions"][dimension],
        }
        for dimension in NORMALIZATION
    }
    summary = {name: {"official": baseline[name], "measured": measured[name], "delta": measured[name] - baseline[name]} for name in ("total", "quality", "semantic")}
    return {"baseline": baseline_name, "summary": summary, "dimensions": dimensions}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_json", type=Path)
    parser.add_argument("--baseline", choices=sorted(OFFICIAL), default="no_prompt")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = comparison(load_raw_scores(args.result_json), args.baseline)
    print("metric                 official   measured      delta")
    for name, values in result["summary"].items():
        print(f"{name:22} {values['official']:8.2f} {values['measured']:10.2f} {values['delta']:+10.2f}")
    print()
    print("dimension              official   measured      delta")
    for name, values in result["dimensions"].items():
        print(f"{name:22} {values['official']:8.2f} {values['measured']:10.2f} {values['delta']:+10.2f}")
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
