import argparse
import importlib
import itertools
import json
from pathlib import Path


def _parse_args():
    parser = argparse.ArgumentParser(description="Build MINERVA video Lance dataset from local metadata and downloaded videos")
    parser.add_argument("--metadata-json", type=Path, required=True, help="Path to minerva.json")
    parser.add_argument("--videos-dir", type=Path, required=True, help="Directory containing downloaded videos")
    parser.add_argument("--output", type=Path, required=True, help="Output Lance directory, e.g. data/train.lance")
    parser.add_argument("--batch-size", type=int, default=8, help="Rows per write batch")
    parser.add_argument("--mode", type=str, default="create", choices=["create", "overwrite", "append"], help="Lance write mode")
    parser.add_argument("--max-rows-per-file", type=int, default=512, help="Maximum rows per data file")
    parser.add_argument("--max-rows-per-group", type=int, default=64, help="Maximum rows per row group")
    parser.add_argument("--max-bytes-per-file-gb", type=int, default=4, help="Maximum file size in GiB")
    parser.add_argument("--data-storage-version", type=str, default="stable", help="Lance storage version, e.g. stable or legacy")
    return parser.parse_args()


def _load_unique_video_ids(metadata_json: Path):
    rows = json.loads(metadata_json.read_text(encoding="utf-8"))
    video_ids = []
    seen = set()
    for row in rows:
        video_id = str(row.get("video_id", "")).strip()
        if video_id and video_id not in seen:
            seen.add(video_id)
            video_ids.append(video_id)
    if not video_ids:
        raise ValueError(f"No valid video_id found in metadata: {metadata_json}")
    return video_ids


def _resolve_video_file(videos_dir: Path, video_id: str):
    for ext in ("mp4", "webm", "mkv", "mov", "MP4", "WEBM", "MKV", "MOV"):
        candidate = videos_dir / f"{video_id}.{ext}"
        if candidate.exists():
            return candidate
    return None


def _build_schema(pa):
    return pa.schema(
        [
            pa.field("video_id", pa.string()),
            pa.field("youtube_url", pa.string()),
            pa.field("video_ext", pa.string()),
            pa.field("video_size_bytes", pa.int64()),
            pa.field("video_blob", pa.large_binary(), metadata={"lance-encoding:blob": "true"}),
        ]
    )


def _batch_iterator(pa, video_ids, videos_dir: Path, batch_size: int, stats: dict, missing_samples: list[str], missing_sample_limit: int):
    idx = 0
    total = len(video_ids)
    while idx < total:
        chunk = video_ids[idx : idx + batch_size]
        cols = {"video_id": [], "youtube_url": [], "video_ext": [], "video_size_bytes": [], "video_blob": []}
        for video_id in chunk:
            stats["scanned"] += 1
            path = _resolve_video_file(videos_dir, video_id)
            if path is None:
                stats["missing"] += 1
                if len(missing_samples) < missing_sample_limit:
                    missing_samples.append(video_id)
                continue
            cols["video_id"].append(video_id)
            cols["youtube_url"].append(f"https://www.youtube.com/watch?v={video_id}")
            cols["video_ext"].append(path.suffix.lower().lstrip("."))
            cols["video_size_bytes"].append(path.stat().st_size)
            cols["video_blob"].append(path.read_bytes())
            stats["written"] += 1

        idx += batch_size
        if cols["video_id"]:
            yield pa.RecordBatch.from_pydict(cols)


def main():
    args = _parse_args()

    lance = importlib.import_module("lance")
    pa = importlib.import_module("pyarrow")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    video_ids = _load_unique_video_ids(args.metadata_json)
    schema = _build_schema(pa)
    stats = {"scanned": 0, "missing": 0, "written": 0}
    missing_samples: list[str] = []
    iterator = _batch_iterator(pa, video_ids, args.videos_dir, args.batch_size, stats, missing_samples, missing_sample_limit=10)

    try:
        first_batch = next(iterator)
    except StopIteration as exc:
        raise ValueError(f"No local videos found under {args.videos_dir}. Checked {len(video_ids)} video_id entries from {args.metadata_json}.") from exc

    lance.write_dataset(
        itertools.chain([first_batch], iterator),
        str(args.output),
        schema=schema,
        mode=args.mode,
        max_rows_per_file=args.max_rows_per_file,
        max_rows_per_group=args.max_rows_per_group,
        max_bytes_per_file=args.max_bytes_per_file_gb * 1024**3,
        data_storage_version=args.data_storage_version,
    )

    print(f"Wrote Lance dataset: {args.output}")
    print(f"Unique video IDs in metadata: {len(video_ids)}")
    print(f"Video IDs scanned: {stats['scanned']}")
    print(f"Video rows written: {stats['written']}")
    print(f"Video IDs missing local file: {stats['missing']}")
    if missing_samples:
        print(f"Missing sample IDs (first {len(missing_samples)}): {', '.join(missing_samples)}")


if __name__ == "__main__":
    main()
