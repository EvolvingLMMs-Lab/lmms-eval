"""Build and optionally publish the Hugging Face VBench prompt dataset.

The source registries are pinned to the VBench revision used immediately before
the official Wan2.2 samples were published. In addition to preserving the
upstream prompt metadata, this builder expands every prompt according to the
official sampling protocol:

* VBench: 5 samples per prompt, 25 for ``temporal_flickering``.
* VBench 2.0: 3 samples per prompt, 20 for ``Diversity``.

Run with uv, for example::

    uv run python lmms_eval/tasks/vbench/build_dataset.py \
        --output-dir /tmp/vbench-hf \
        --repo-id pufanyi/VBench \
        --push-to-hub
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any
from urllib.request import urlopen

SOURCE_REVISION = "1ee42dada7a2f7cfaf4290e8a02d087f6f8ee425"
SOURCE_ROOT = f"https://raw.githubusercontent.com/Vchitect/VBench/{SOURCE_REVISION}"
SOURCES = {
    "vbench": f"{SOURCE_ROOT}/vbench/VBench_full_info.json",
    "vbench2": f"{SOURCE_ROOT}/VBench-2.0/vbench2/VBench2_full_info.json",
}
LICENSE_URL = f"{SOURCE_ROOT}/LICENSE"


def download_bytes(url: str) -> bytes:
    with urlopen(url) as response:
        return response.read()


def parse_records(payload: bytes, source: str) -> list[dict[str, Any]]:
    data = json.loads(payload)
    if not isinstance(data, list) or not all(isinstance(record, dict) for record in data):
        raise ValueError(f"Expected a list of objects from {source}")
    return data


def normalize_dimension(name: str) -> str:
    """Match the task names already used by the VBench integration."""

    return name.strip().lower().replace(" ", "_")


def prompt_id(suite: str, prompt: str) -> str:
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    return f"{suite}-{digest}"


def deterministic_seed(suite: str, prompt: str, sample_index: int) -> int:
    """Return a stable per-sample seed without claiming to match private official seeds."""

    payload = f"{suite}\0{prompt}\0{sample_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def sample_count(suite: str, dimensions: list[str]) -> int:
    if suite == "vbench":
        return 25 if "temporal_flickering" in dimensions else 5
    if suite == "vbench2":
        return 20 if "diversity" in dimensions else 3
    raise ValueError(f"Unknown suite: {suite}")


def merge_prompts(records: list[dict[str, Any]], suite: str) -> list[dict[str, Any]]:
    """Merge duplicate prompt strings while retaining all dimension metadata."""

    merged: dict[str, dict[str, Any]] = {}
    for source_record_id, record in enumerate(records):
        prompt = record.get("prompt_en")
        dimensions = record.get("dimension", [])
        if isinstance(dimensions, str):
            dimensions = [dimensions]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Missing prompt_en in {suite} record {source_record_id}")
        if not isinstance(dimensions, list) or not dimensions:
            raise ValueError(f"Missing dimension in {suite} record {source_record_id}")

        item = merged.setdefault(
            prompt,
            {
                "prompt": prompt,
                "dimensions": [],
                "official_dimensions": [],
                "source_record_ids": [],
                "auxiliary": [],
            },
        )
        for official_dimension in dimensions:
            if not isinstance(official_dimension, str) or not official_dimension.strip():
                raise ValueError(f"Invalid dimension in {suite} record {source_record_id}")
            normalized = normalize_dimension(official_dimension)
            if normalized not in item["dimensions"]:
                item["dimensions"].append(normalized)
                item["official_dimensions"].append(official_dimension)
        item["source_record_ids"].append(source_record_id)
        if "auxiliary_info" in record:
            item["auxiliary"].append(
                {
                    "source_record_id": source_record_id,
                    "dimensions": [normalize_dimension(value) for value in dimensions],
                    "value": record["auxiliary_info"],
                }
            )
    return list(merged.values())


def expand_samples(suite: str, prompts: list[dict[str, Any]], dimension: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in prompts:
        if dimension is not None and dimension not in item["dimensions"]:
            continue
        row_dimensions = [dimension] if dimension is not None else item["dimensions"]
        official_dimensions = [official for normalized, official in zip(item["dimensions"], item["official_dimensions"]) if dimension is None or normalized == dimension]
        count = sample_count(suite, row_dimensions)
        pid = prompt_id(suite, item["prompt"])
        for sample_index in range(count):
            rows.append(
                {
                    "id": f"{pid}-{sample_index:02d}",
                    "suite": suite,
                    "prompt_id": pid,
                    "prompt": item["prompt"],
                    "dimensions": row_dimensions,
                    "official_dimensions": official_dimensions,
                    "auxiliary_info": json.dumps(item["auxiliary"], ensure_ascii=False, sort_keys=True),
                    "sample_index": sample_index,
                    "num_samples": count,
                    "seed": deterministic_seed(suite, item["prompt"], sample_index),
                    "source_record_ids": item["source_record_ids"],
                }
            )
    return rows


def build_configs(source_records: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    configs: dict[str, list[dict[str, Any]]] = {}
    for suite, records in source_records.items():
        prompts = merge_prompts(records, suite)
        configs[suite] = expand_samples(suite, prompts)
        dimensions = sorted({dimension for item in prompts for dimension in item["dimensions"]})
        for dimension in dimensions:
            configs[f"{suite}_{dimension}"] = expand_samples(suite, prompts, dimension)
    return configs


def dataset_features():
    from datasets import Features, Sequence, Value

    return Features(
        {
            "id": Value("string"),
            "suite": Value("string"),
            "prompt_id": Value("string"),
            "prompt": Value("string"),
            "dimensions": Sequence(Value("string")),
            "official_dimensions": Sequence(Value("string")),
            "auxiliary_info": Value("string"),
            "sample_index": Value("int32"),
            "num_samples": Value("int32"),
            "seed": Value("int64"),
            "source_record_ids": Sequence(Value("int32")),
        }
    )


def dataset_card(configs: dict[str, list[dict[str, Any]]]) -> str:
    config_yaml = "\n".join(f"  - config_name: {name}\n    data_files:\n      - split: test\n        path: data/{name}/test-*.parquet" for name in sorted(configs))
    counts = "\n".join(f"| `{name}` | {len(rows):,} |" for name, rows in sorted(configs.items()))
    return f"""---
license: apache-2.0
language:
  - en
task_categories:
  - text-to-video
pretty_name: VBench and VBench 2.0 Prompt Suites
configs:
{config_yaml}
---

# VBench prompt suites for lmms-eval

This is a Hugging Face-native conversion of the official [VBench](https://github.com/Vchitect/VBench) and VBench 2.0 prompt registries. The source is pinned to [`{SOURCE_REVISION}`](https://github.com/Vchitect/VBench/tree/{SOURCE_REVISION}), the VBench revision immediately preceding the official Wan2.2 sample release.

Each row represents one video-generation request. Rows are expanded according to the official protocol: VBench uses 5 samples per prompt and 25 for `temporal_flickering`; VBench 2.0 uses 3 samples per prompt and 20 for `Diversity`. Exact duplicate prompt strings are merged in the combined `vbench` and `vbench2` configs so a generated video can be reused across dimensions.

The `seed` column is a deterministic lmms-eval seed derived from suite, prompt, and sample index. VBench does not prescribe exact seed values, and the official Wan2.2 seeds were not published; these seeds provide reproducibility but do not claim bit-for-bit equivalence with official samples.

## Config sizes

| Config | Rows |
|---|---:|
{counts}

## Columns

- `prompt`: text passed to the video generator.
- `dimensions` / `official_dimensions`: normalized lmms-eval names and upstream names.
- `sample_index`, `num_samples`, `seed`: sampling metadata.
- `auxiliary_info`: JSON-encoded upstream metadata used by VBench scorers.
- `source_record_ids`: indices into the pinned upstream full-info registry.

The unmodified source registries and the upstream Apache-2.0 license are included under `source/` and `LICENSE`.
"""


def write_dataset(output_dir: Path, configs: dict[str, list[dict[str, Any]]], source_payloads: dict[str, bytes]) -> None:
    from datasets import Dataset

    output_dir.mkdir(parents=True, exist_ok=True)
    features = dataset_features()
    for config_name, rows in sorted(configs.items()):
        config_dir = output_dir / "data" / config_name
        config_dir.mkdir(parents=True, exist_ok=True)
        Dataset.from_list(rows, features=features).to_parquet(config_dir / "test-00000-of-00001.parquet")

    source_dir = output_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    for suite, payload in source_payloads.items():
        filename = "VBench_full_info.json" if suite == "vbench" else "VBench2_full_info.json"
        (source_dir / filename).write_bytes(payload)
    (output_dir / "README.md").write_text(dataset_card(configs), encoding="utf-8")
    (output_dir / "LICENSE").write_bytes(download_bytes(LICENSE_URL))


def push_dataset(output_dir: Path, repo_id: str, private: bool) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    commit = api.upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Add VBench prompt suites from {SOURCE_REVISION[:12]}",
    )
    return commit.oid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default="pufanyi/VBench")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} exists; pass --overwrite to replace it")
        shutil.rmtree(args.output_dir)

    source_payloads = {suite: download_bytes(url) for suite, url in SOURCES.items()}
    source_records = {suite: parse_records(payload, SOURCES[suite]) for suite, payload in source_payloads.items()}
    configs = build_configs(source_records)
    write_dataset(args.output_dir, configs, source_payloads)

    print(f"Built {len(configs)} configs in {args.output_dir}")
    for name, rows in sorted(configs.items()):
        print(f"- {name}: {len(rows)} rows")
    if args.push_to_hub:
        commit = push_dataset(args.output_dir, args.repo_id, args.private)
        print(f"Published https://huggingface.co/datasets/{args.repo_id}/tree/{commit}")


if __name__ == "__main__":
    main()
