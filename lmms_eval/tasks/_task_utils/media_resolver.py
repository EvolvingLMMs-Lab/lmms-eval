import os
from pathlib import Path

_VIDEO_EXTENSIONS = ("mp4", "webm", "mkv", "mov", "avi", "flv", "wmv")
_AUDIO_EXTENSIONS = ("wav", "mp3", "m4a", "aac", "flac", "ogg", "opus", "webm")


def _dedupe(items):
    seen = set()
    deduped = []
    for item in items:
        key = str(item)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _hf_home_path():
    return Path(os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface")))


def _candidate_roots(cache_dir=None, env_vars=(), media_type="video"):
    roots = []

    for env_name in env_vars:
        env_value = os.getenv(env_name, "").strip()
        if env_value:
            roots.append(Path(os.path.expanduser(env_value)))

    global_media_root = os.getenv("LMMS_EVAL_MEDIA_ROOT", "").strip()
    if global_media_root:
        roots.append(Path(os.path.expanduser(global_media_root)))

    hf_home = _hf_home_path()
    roots.append(hf_home)

    if cache_dir:
        cache_root = hf_home / cache_dir
        roots.append(cache_root)
        roots.append(hf_home / "datasets" / cache_dir)

        if media_type == "audio":
            roots.append(cache_root / "audio")
            roots.append(cache_root / "audios")
            roots.append(cache_root / "clips")
            roots.append((hf_home / "datasets" / cache_dir) / "audio")
            roots.append((hf_home / "datasets" / cache_dir) / "audios")
        else:
            roots.append(cache_root / "video")
            roots.append(cache_root / "videos")
            roots.append(cache_root / "clips")
            roots.append((hf_home / "datasets" / cache_dir) / "video")
            roots.append((hf_home / "datasets" / cache_dir) / "videos")

    return _dedupe(roots)


def _extension_variants(path, media_type="video"):
    variants = []
    suffix = path.suffix
    if suffix:
        variants.append(path)
        variants.append(path.with_suffix(suffix.lower()))
        variants.append(path.with_suffix(suffix.upper()))
        return _dedupe(variants)

    extensions = _AUDIO_EXTENSIONS if media_type == "audio" else _VIDEO_EXTENSIONS
    for ext in extensions:
        variants.append(path.with_suffix(f".{ext}"))
        variants.append(path.with_suffix(f".{ext.upper()}"))
    return _dedupe(variants)


def resolve_media_reference(reference, media_type="video", cache_dir=None, env_vars=(), extra_subdirs=()):
    if not isinstance(reference, str):
        return reference

    value = reference.strip()
    if not value:
        return reference

    lowered = value.lower()
    if lowered.startswith("http://") or lowered.startswith("https://"):
        return reference

    expanded = Path(os.path.expanduser(value))
    if expanded.exists():
        return str(expanded)

    direct = Path(value)
    if direct.exists():
        return str(direct)

    basename = os.path.basename(value)
    candidates = []

    for root in _candidate_roots(cache_dir=cache_dir, env_vars=env_vars, media_type=media_type):
        for subdir in ("", *extra_subdirs):
            parent = root / subdir if subdir else root
            candidates.extend(_extension_variants(parent / value, media_type=media_type))
            candidates.extend(_extension_variants(parent / basename, media_type=media_type))

    for candidate in _dedupe(candidates):
        if candidate.exists():
            return str(candidate)

    return reference
