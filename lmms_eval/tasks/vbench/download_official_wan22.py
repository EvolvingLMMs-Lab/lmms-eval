"""Download the official Wan2.2 VBench samples with byte-level validation.

Google Drive can end a large response early while ``gdown`` still renames the
temporary file to its final name. The official archive consists of eleven
4-GiB split ZIP volumes plus the final ZIP segment, so accepting such a file
silently produces an unusable archive. This downloader resumes by HTTP range,
checks ``Content-Range``, and only reports completion after both the published
size and Drive SHA-256 digest match.

Run with uv, for example::

    uv run --no-project --with 'gdown>=5.2,<6' python \
        lmms_eval/tasks/vbench/download_official_wan22.py \
        --output-dir /data/Wan2.2-T2V-A14B-wo-prompt-extend
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

FOUR_GIB = 4_294_967_296
FILES = {
    "Wan2.2-T2V-A14B.z01": ("1-R3tkQsrLWwyo9azHddNmnEgOb8f2I3x", FOUR_GIB, "f106add6e9ea09a289000d12775bff5189cdfddb8608718fb07759ee69866f45"),
    "Wan2.2-T2V-A14B.z02": ("1jdLboTjwTWghYT7BF5jWVq9Ip0s9hl0E", FOUR_GIB, "f002525a732bb67fea346bf46ee3e438bf82d44417d98087bbd54fbf644245c2"),
    "Wan2.2-T2V-A14B.z03": ("19QXAE1MN84vlkchzyShAN-2OfvTPl0xw", FOUR_GIB, "81a2530d886ad90acb71dd91f0c3a555cd878b26bda9867a2967e0a492f34fef"),
    "Wan2.2-T2V-A14B.z04": ("1XlUm3RFkkiQncEvZ2DxMe5SzqhkounXP", FOUR_GIB, "e79b3c91af7c55f59486f528f3f2d8dbaa9375aac08568bf47464faea342507d"),
    "Wan2.2-T2V-A14B.z05": ("13rvO_8ltj5S9s1CMT5QgswkHcynTntJg", FOUR_GIB, "6943c0d871ea2d9128b400c23659866c487d2ca84957422326d015b7996f2046"),
    "Wan2.2-T2V-A14B.z06": ("1X1LveYzboTEpPmh_9hMutjQdAzz1S1wi", FOUR_GIB, "83a40bea828edf2af8ee526c89d23b0869a8f34143d747a281a91a455f124fe0"),
    "Wan2.2-T2V-A14B.z07": ("1Zpm0Wm9kdf-Ep0S69LhFrszE2vKCUmLX", FOUR_GIB, "d380c33a32913f1a279abe2e19adf914647b228f8a2ea3dd117b5d01b9dbb26b"),
    "Wan2.2-T2V-A14B.z08": ("19-201M3OGE2t22Nd98UnzyP_w5hQpNVx", FOUR_GIB, "77649e1d8efeb8b3b7df34ae3eee474e1ef0433e9206bbf6fbd3f0eb531a7028"),
    "Wan2.2-T2V-A14B.z09": ("1XrJky79Mmm1u2nBRU8DiZX47mBf89U9s", FOUR_GIB, "de6a60a0f4c32aea2b8b2c5ede29438bb8f274e6632c68e4f1fa8b6fbef69b5f"),
    "Wan2.2-T2V-A14B.z10": ("1j21e7xyMOIJ6qMMahc6Jk6-f96-hik8g", FOUR_GIB, "74ecd178ebebb9daa7d6afb0c3ae9c27f32ddbb142450e28eb9eb6ad38974486"),
    "Wan2.2-T2V-A14B.z11": ("12x2SNA-0BkI59O0jfF4to5fCAHd5wv86", FOUR_GIB, "47c51d7fdda1ce1c69bb04ab2312a5fea208b83d4ba47133100cb2dcecef3792"),
    "Wan2.2-T2V-A14B.zip": ("1ldurYzYSgTs0vjxrjf1DbuwhKBTEo3m4", 2_622_543_855, "43ae5ff914d66c07d8b28ee08006309c1fc3317e19464ca62a9de7687a7f85ad"),
}
CONTENT_RANGE_RE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")


def work_path(output_dir: Path, filename: str) -> Path:
    target = output_dir / filename
    if target.exists():
        return target
    partials = list(output_dir.glob(f"{filename}*.part"))
    if len(partials) > 1:
        raise RuntimeError(f"Multiple partial files found for {filename}: {partials}")
    return partials[0] if partials else output_dir / f"{filename}.download.part"


def open_drive_response(file_id: str, start: int):
    """Resolve a public Drive URL and return a validated ranged response."""

    download_module = importlib.import_module("gdown.download")
    session, _cookies_file = download_module._get_session(
        proxy=None,
        use_cookies=True,
        user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        return_cookies_file=True,
    )
    url = f"https://drive.google.com/uc?id={file_id}"
    response = None
    try:
        while True:
            response = session.get(url, stream=True, timeout=60)
            if "Content-Disposition" in response.headers:
                break
            body = response.text
            response.close()
            url = download_module.get_url_from_gdrive_confirmation(body)

        if start:
            response.close()
            response = session.get(url, headers={"Range": f"bytes={start}-"}, stream=True, timeout=60)
            content_range = response.headers.get("Content-Range", "")
            match = CONTENT_RANGE_RE.fullmatch(content_range)
            if response.status_code != 206 or match is None or int(match.group(1)) != start:
                raise RuntimeError(f"Drive ignored range {start}: status={response.status_code}, Content-Range={content_range!r}")
        return session, response
    except Exception:
        if response is not None:
            response.close()
        session.close()
        raise


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(output_dir: Path, filename: str, file_id: str, expected_size: int, expected_sha256: str, max_retries: int) -> tuple[str, int]:
    path = work_path(output_dir, filename)
    current_size = path.stat().st_size if path.exists() else 0
    if current_size > expected_size:
        raise RuntimeError(f"{path} is larger than expected ({current_size} > {expected_size}); move it aside and retry")

    failures = 0
    while current_size < expected_size:
        session = response = None
        try:
            session, response = open_drive_response(file_id, current_size)
            content_range = response.headers.get("Content-Range")
            if content_range:
                match = CONTENT_RANGE_RE.fullmatch(content_range)
                if match is None or int(match.group(3)) != expected_size:
                    raise RuntimeError(f"Unexpected Content-Range for {filename}: {content_range!r}")
            elif current_size:
                raise RuntimeError(f"Missing Content-Range while resuming {filename} at {current_size}")

            with path.open("ab") as output:
                for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        output.write(chunk)
            current_size = path.stat().st_size
            if current_size > expected_size:
                raise RuntimeError(f"{filename} exceeded its expected size ({current_size} > {expected_size})")
            failures = 0
            print(f"progress {filename} {current_size}/{expected_size}", flush=True)
        except Exception as exc:
            current_size = path.stat().st_size if path.exists() else 0
            failures += 1
            if failures > max_retries:
                raise RuntimeError(f"Giving up on {filename} at {current_size}/{expected_size}") from exc
            delay = min(60, 2 ** min(failures, 6))
            print(f"retry {filename} at {current_size}/{expected_size} in {delay}s: {exc}", flush=True)
            time.sleep(delay)
        finally:
            if response is not None:
                response.close()
            if session is not None:
                session.close()

    target = output_dir / filename
    if path != target:
        path.replace(target)
    actual_sha256 = sha256(target)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(f"SHA-256 mismatch for {target}: actual={actual_sha256}, expected={expected_sha256}; move the corrupt file aside and retry")
    return filename, expected_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=100)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.verify_only:
        invalid = []
        for filename, (_file_id, expected_size, expected_sha256) in FILES.items():
            path = args.output_dir / filename
            actual_size = path.stat().st_size if path.exists() else None
            actual_sha256 = sha256(path) if actual_size == expected_size else None
            print(f"{filename}: size={actual_size}/{expected_size}, sha256={actual_sha256}/{expected_sha256}")
            if actual_size != expected_size or actual_sha256 != expected_sha256:
                invalid.append(filename)
        if invalid:
            raise SystemExit(f"Incomplete or invalid files: {', '.join(invalid)}")
        return

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_file, args.output_dir, filename, file_id, expected_size, expected_sha256, args.max_retries) for filename, (file_id, expected_size, expected_sha256) in FILES.items()]
        for future in as_completed(futures):
            filename, size = future.result()
            print(f"completed {filename} {size}", flush=True)


if __name__ == "__main__":
    main()
