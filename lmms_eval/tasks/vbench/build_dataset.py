from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.request import urlopen

V1_URL = "https://raw.githubusercontent.com/Vchitect/VBench/master/vbench/VBench_full_info.json"
V2_URL = "https://raw.githubusercontent.com/Vchitect/VBench/master/VBench-2.0/vbench2/VBench2_full_info.json"


def download_json(url: str) -> list[dict[str, Any]]:
    with urlopen(url) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        payload = response.read().decode(charset)
    data = json.loads(payload)
    if not isinstance(data, list):
        raise ValueError(f"Expected list from {url}, got {type(data).__name__}")
    return data


def normalize_dimension(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def to_prompt_row(record: dict[str, Any]) -> dict[str, Any] | None:
    prompt = record.get("prompt_en")
    if not isinstance(prompt, str) or not prompt.strip():
        return None

    dimensions = record.get("dimension", [])
    if isinstance(dimensions, str):
        dimensions = [dimensions]
    if not isinstance(dimensions, list):
        return None

    normalized_dimensions = [normalize_dimension(dim) for dim in dimensions if isinstance(dim, str) and dim.strip()]
    if not normalized_dimensions:
        return None

    return {
        "prompt": prompt,
        "dimension": normalized_dimensions,
        "auxiliary_info": record.get("auxiliary_info"),
    }


def to_dimension_rows(prompt_row: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dim in prompt_row["dimension"]:
        rows.append(
            {
                "prompt": prompt_row["prompt"],
                "dimension": dim,
                "auxiliary_info": prompt_row["auxiliary_info"],
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False))
            file.write("\n")


def build_dimension_files(records: list[dict[str, Any]], data_dir: Path, all_filename: str) -> tuple[dict[str, int], int]:
    by_dimension: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_prompt_rows: list[dict[str, Any]] = []

    for record in records:
        prompt_row = to_prompt_row(record)
        if prompt_row is None:
            continue
        rows = to_dimension_rows(prompt_row)
        for row in rows:
            by_dimension[row["dimension"]].append(row)
        all_prompt_rows.append(prompt_row)

    for dimension, rows in sorted(by_dimension.items()):
        write_jsonl(data_dir / f"{dimension}.jsonl", rows)

    write_jsonl(data_dir / all_filename, all_prompt_rows)

    return {dimension: len(rows) for dimension, rows in sorted(by_dimension.items())}, len(all_prompt_rows)


def print_summary(title: str, counts: dict[str, int], total_prompts: int) -> None:
    print(f"\n{title}")
    print(f"- dimensions: {len(counts)}")
    print(f"- prompts: {total_prompts}")
    for dimension, count in counts.items():
        print(f"  - {dimension}: {count}")


def main() -> None:
    task_dir = Path(__file__).resolve().parent
    data_dir = task_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    v1_records = download_json(V1_URL)
    v2_records = download_json(V2_URL)

    v1_counts, v1_prompt_total = build_dimension_files(v1_records, data_dir, "all_dimensions.jsonl")
    v2_counts, v2_prompt_total = build_dimension_files(v2_records, data_dir, "all_dimensions_v2.jsonl")

    print_summary("VBench v1", v1_counts, v1_prompt_total)
    print_summary("VBench v2", v2_counts, v2_prompt_total)


if __name__ == "__main__":
    main()
