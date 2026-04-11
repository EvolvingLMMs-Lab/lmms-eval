import os
import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from lmms_eval.tasks._task_utils.media_resolver import resolve_media_reference
from lmms_eval.utils import eval_logger

_VIDEO_EXTENSIONS = ("mp4", "MP4", "mkv", "webm", "mov")

# ---------------------------------------------------------------------------
# Lazy S3 extraction — pull trimmed segments on demand
# ---------------------------------------------------------------------------
_S3_INDEX = None  # populated on first miss
_S3_CLIENT = None


def _parse_video_ref(ref: str):
    """Parse 'uuid_start_end' → (uid, start, end) or None."""
    parts = ref.rsplit("_", 2)
    if len(parts) != 3:
        return None
    uid, start_s, end_s = parts
    try:
        return uid, float(start_s), float(end_s)
    except ValueError:
        return None


def _get_s3_client():
    global _S3_CLIENT
    if _S3_CLIENT is not None:
        return _S3_CLIENT
    try:
        import boto3

        creds_path = os.path.expanduser("~/.aws/credentials")
        cfg_path = os.path.expanduser("~/.aws/config")
        if not os.path.exists(creds_path):
            return None
        creds = open(creds_path).read()
        cfg = open(cfg_path).read() if os.path.exists(cfg_path) else ""
        endpoint = "https://storage.eu-north1.nebius.cloud:443"
        for line in cfg.split("\n"):
            if "endpoint_url" in line:
                endpoint = line.split("=", 1)[1].strip()
                break
        _S3_CLIENT = boto3.client(
            "s3",
            endpoint_url=endpoint,
            region_name="eu-north1",
            aws_access_key_id=creds.split("aws_access_key_id = ")[1].split("\n")[0].strip(),
            aws_secret_access_key=creds.split("aws_secret_access_key = ")[1].split("\n")[0].strip(),
        )
        return _S3_CLIENT
    except Exception as e:
        eval_logger.warning("S3 client init failed: {}", e)
        return None


def _get_s3_index():
    """Build uid → {tar_path, offset, size} from the Lance index."""
    global _S3_INDEX
    if _S3_INDEX is not None:
        return _S3_INDEX
    _S3_INDEX = {}
    try:
        import lance

        creds = open(os.path.expanduser("~/.aws/credentials")).read()
        opts = {
            "aws_endpoint": "https://storage.eu-north1.nebius.cloud:443",
            "aws_region": "eu-north1",
            "aws_access_key_id": creds.split("aws_access_key_id = ")[1].split("\n")[0].strip(),
            "aws_secret_access_key": creds.split("aws_secret_access_key = ")[1].split("\n")[0].strip(),
        }
        ds = lance.dataset(
            "s3://ami-labs-data/public/tables/s3_video_primary.lance",
            storage_options=opts,
        )
        table = ds.to_table(
            filter="source = 'ego4d' AND member_path LIKE 'ego4d_full_scale/%'",
            columns=["member_path", "tar_path", "offset", "size"],
        )
        for i in range(table.num_rows):
            mp = table.column("member_path")[i].as_py()
            uid = mp.replace("ego4d_full_scale/", "").replace(".mp4", "")
            _S3_INDEX[uid] = {
                "tar_path": table.column("tar_path")[i].as_py(),
                "offset": table.column("offset")[i].as_py(),
                "size": table.column("size")[i].as_py(),
            }
        eval_logger.info("EgoTempo S3 index: {} ego4d clips available", len(_S3_INDEX))
    except Exception as e:
        eval_logger.warning("Failed to load S3 index: {}", e)
    return _S3_INDEX


def _extract_trimmed_from_s3(uid: str, start: float, end: float, out_path: str) -> bool:
    """Extract a trimmed video segment from an S3 tar and save to out_path."""
    s3 = _get_s3_client()
    index = _get_s3_index()
    if s3 is None or uid not in index:
        return False

    info = index[uid]
    bucket = "ami-labs-data"
    tar_key = info["tar_path"]
    offset = info["offset"]
    size = info["size"]

    try:
        # Read tar header to find data offset
        header_resp = s3.get_object(Bucket=bucket, Key=tar_key, Range=f"bytes={offset}-{offset + 511}")
        header = header_resp["Body"].read()
        magic = header[257:262]
        if magic == b"ustar":
            data_offset = offset + 512
            # Read actual size from tar header (octal, bytes 124-136)
            file_size_raw = header[124:136].rstrip(b"\x00").rstrip()
            file_size = int(file_size_raw, 8) if file_size_raw else size
        else:
            data_offset = offset
            file_size = size

        # Stream full clip to temp file, trim with ffmpeg -c copy
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
            data_range = f"bytes={data_offset}-{data_offset + file_size - 1}"
            resp = s3.get_object(Bucket=bucket, Key=tar_key, Range=data_range)
            # Stream in chunks to avoid loading full clip in memory
            for chunk in resp["Body"].iter_chunks(chunk_size=8 * 1024 * 1024):
                tmp.write(chunk)
            tmp.flush()

            # ffmpeg trim: -c copy is fast (no re-encode), output is small
            cmd = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-ss",
                f"{start:.3f}",
                "-to",
                f"{end:.3f}",
                "-i",
                tmp.name,
                "-c",
                "copy",
                out_path,
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=120)
            if result.returncode != 0:
                eval_logger.warning("ffmpeg trim failed for {}: {}", uid, result.stderr.decode()[:200])
                return False

        # Validate output — ffmpeg succeeds but produces empty file when
        # the requested time range exceeds the source video duration.
        if not os.path.exists(out_path) or os.path.getsize(out_path) < 1024:
            if os.path.exists(out_path):
                os.remove(out_path)
            eval_logger.warning(
                "EgoTempo: trim produced empty output for {} (segment {:.1f}-{:.1f}s " "likely exceeds video duration)",
                uid,
                start,
                end,
            )
            return False

        eval_logger.info("EgoTempo: extracted {:.0f}s segment for {} ({})", end - start, uid, out_path)
        return True
    except Exception as e:
        eval_logger.warning("S3 extraction failed for {}: {}", uid, e)
        return False


def _normalize_text(text: Any) -> str:
    if text is None:
        return ""
    normalized = str(text).strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    if len(left) > len(right):
        left, right = right, left

    previous = list(range(len(left) + 1))
    for i, right_ch in enumerate(right, start=1):
        current = [i]
        for j, left_ch in enumerate(left, start=1):
            insertion = previous[j] + 1
            deletion = current[j - 1] + 1
            substitution = previous[j - 1] + (left_ch != right_ch)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def _anls_score(prediction: str, answer: str, threshold: float = 0.5) -> float:
    pred = _normalize_text(prediction)
    target = _normalize_text(answer)
    if not pred and not target:
        return 1.0
    if not pred or not target:
        return 0.0

    distance = _levenshtein_distance(pred, target)
    normalized_distance = distance / max(len(pred), len(target))
    score = 1.0 - normalized_distance
    if score < threshold:
        return 0.0
    return score


def _strip_answer_prefix(text: str) -> str:
    cleaned = str(text).strip()
    prefixes = [
        "the answer is",
        "answer:",
        "the correct answer is",
        "the final answer is",
    ]

    lowered = cleaned.lower()
    for prefix in prefixes:
        if lowered.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip(" :.-")
            break
    return cleaned


def _candidate_video_dirs() -> list[Path]:
    paths = []

    explicit_video_dir = os.getenv("EGOTEMPO_VIDEO_DIR", "").strip()
    if explicit_video_dir:
        paths.append(Path(os.path.expanduser(explicit_video_dir)))

    explicit_cache_dir = os.getenv("EGOTEMPO_CACHE_DIR", "").strip()
    if explicit_cache_dir:
        paths.append(Path(os.path.expanduser(explicit_cache_dir)))

    hf_home = Path(os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface/")))
    paths.append(hf_home / "egotempo")

    deduped = []
    seen = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _resolve_video_path(clip_id: str) -> str | None:
    if clip_id == "":
        return None

    resolved = resolve_media_reference(clip_id, media_type="video", cache_dir="egotempo", env_vars=("EGOTEMPO_VIDEO_DIR", "EGOTEMPO_CACHE_DIR"))
    if isinstance(resolved, str) and os.path.exists(resolved):
        return resolved

    for root in _candidate_video_dirs():
        for extension in _VIDEO_EXTENSIONS:
            candidate = root / f"{clip_id}.{extension}"
            if candidate.exists():
                return str(candidate)
    return None


def egotempo_doc_to_visual(doc):
    video_ref = str(doc.get("video", "")).strip()

    # 1. Try to find a pre-existing file (pre-trimmed or full clip)
    if video_ref:
        resolved = resolve_media_reference(
            video_ref,
            media_type="video",
            cache_dir="egotempo",
            env_vars=("EGOTEMPO_VIDEO_DIR", "EGOTEMPO_CACHE_DIR"),
        )
        if isinstance(resolved, str) and os.path.exists(resolved):
            return [resolved]

    # 2. Check other doc keys
    for key in ["video_path", "media_path", "clip_path", "path", "file"]:
        value = doc.get(key)
        if value:
            resolved = resolve_media_reference(
                value,
                media_type="video",
                cache_dir="egotempo",
                env_vars=("EGOTEMPO_VIDEO_DIR", "EGOTEMPO_CACHE_DIR"),
            )
            if isinstance(resolved, str) and os.path.exists(resolved):
                return [resolved]

    # 3. Try full clip by UID (without timestamp suffix)
    parsed = _parse_video_ref(video_ref) if video_ref else None
    if parsed:
        uid, start, end = parsed
        full_clip = _resolve_video_path(uid)
        if full_clip:
            # Return dict with temporal range — our media module handles slicing
            return [{"type": "video", "path": full_clip, "video_start": start, "video_end": end}]

    # 4. Lazy S3 extraction — download full clip from tar, trim, cache
    if parsed:
        uid, start, end = parsed
        cache_dir = os.getenv("EGOTEMPO_CACHE_DIR", "")
        if not cache_dir:
            hf_home = os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface"))
            cache_dir = os.path.join(hf_home, "egotempo")
        out_path = os.path.join(cache_dir, f"{video_ref}.mp4")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return [out_path]
        if _extract_trimmed_from_s3(uid, start, end, out_path):
            return [out_path]

    # 5. Last resort: clip_id fallback
    clip_id = str(doc.get("clip_id", "")).strip()
    video_path = _resolve_video_path(clip_id)
    if video_path is None:
        eval_logger.warning("EgoTempo: no video found for ref={}", video_ref)
        return []
    return [video_path]


def egotempo_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    question = str(doc.get("question", "")).strip()
    return f"{pre_prompt}{question}{post_prompt}"


def egotempo_doc_to_target(doc):
    return str(doc.get("answer", "")).strip()


def egotempo_process_results(doc, results):
    prediction = _strip_answer_prefix(results[0] if results else "")
    answer = str(doc.get("answer", "")).strip()
    score = _anls_score(prediction, answer)

    return {
        "egotempo_anls": {
            "score": score,
            "question_type": str(doc.get("question_type", "unknown")),
        }
    }


def egotempo_aggregate_results(items):
    if not items:
        return 0.0

    total_score = 0.0
    by_category = defaultdict(list)

    for item in items:
        score = float(item.get("score", 0.0))
        category = str(item.get("question_type", "unknown"))
        total_score += score
        by_category[category].append(score)

    for category in sorted(by_category):
        scores = by_category[category]
        category_score = sum(scores) / len(scores)
        eval_logger.info("EgoTempo [{}] ANLS: {:.2f}", category, category_score * 100)

    overall = total_score / len(items)
    eval_logger.info("EgoTempo overall ANLS: {:.2f}", overall * 100)
    return overall
