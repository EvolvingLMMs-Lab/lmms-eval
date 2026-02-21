import argparse
import json
from pathlib import Path


def _parse_args():
    parser = argparse.ArgumentParser(description="Build MINERVA video Lance dataset from local metadata and downloaded videos")
    parser.add_argument("--metadata-json", type=Path, required=True, help="Path to minerva.json")
    parser.add_argument("--videos-dir", type=Path, required=True, help="Directory containing downloaded videos")
    parser.add_argument("--output", type=Path, required=True, help="Output Lance directory, e.g. data/train.lance")
    parser.add_argument("--batch-size", type=int, default=8, help="Rows per write batch")
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


def _batch_iterator(pa, video_ids, videos_dir: Path, batch_size: int):
    idx = 0
    total = len(video_ids)
    while idx < total:
        chunk = video_ids[idx : idx + batch_size]
        cols = {"video_id": [], "youtube_url": [], "video_ext": [], "video_size_bytes": [], "video_blob": []}
        for video_id in chunk:
            path = _resolve_video_file(videos_dir, video_id)
            if path is None:
                continue
            cols["video_id"].append(video_id)
            cols["youtube_url"].append(f"https://www.youtube.com/watch?v={video_id}")
            cols["video_ext"].append(path.suffix.lower().lstrip("."))
            cols["video_size_bytes"].append(path.stat().st_size)
            cols["video_blob"].append(path.read_bytes())

        idx += batch_size
        if cols["video_id"]:
            yield pa.RecordBatch.from_pydict(cols)


def main():
    args = _parse_args()

    import lance
    import pyarrow as pa

    args.output.parent.mkdir(parents=True, exist_ok=True)

    video_ids = _load_unique_video_ids(args.metadata_json)
    schema = _build_schema(pa)
    iterator = _batch_iterator(pa, video_ids, args.videos_dir, args.batch_size)

    lance.write_dataset(iterator, str(args.output), schema=schema, mode="create")

    print(f"Wrote Lance dataset: {args.output}")
    print(f"Unique video IDs in metadata: {len(video_ids)}")


if __name__ == "__main__":
    main()
