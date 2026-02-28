import os
import re
import threading
from pathlib import Path

from lmms_eval.tasks._task_utils.lance_video_resolver import LanceVideoBlobResolver
from lmms_eval.tasks._task_utils.media_resolver import resolve_media_reference
from lmms_eval.utils import eval_logger

_VIDEO_EXTENSIONS = ("mp4", "webm", "mkv", "mov")
_LANCE_RESOLVERS = {}
_LANCE_RESOLVER_LOCK = threading.Lock()


def _dedupe_strings(values):
    seen = set()
    result = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _split_lance_uris(value):
    if value is None:
        return []
    return _dedupe_strings(str(value).replace("\n", ",").split(","))


def _configured_lance_uris():
    uris = []
    single_uri = os.getenv("COUNTIX_LANCE_VIDEO_URI", "").strip()
    if single_uri:
        uris.append(single_uri)
    uris.extend(_split_lance_uris(os.getenv("COUNTIX_LANCE_VIDEO_URIS", "")))
    return _dedupe_strings(uris)


def _candidate_lance_uris(doc):
    uris = []
    for key in ("lance_uri", "video_lance_uri"):
        value = doc.get(key)
        if value:
            uris.extend(_split_lance_uris(value))
    uris.extend(_configured_lance_uris())
    return _dedupe_strings(uris)


def _get_lance_resolver(dataset_uri):
    resolver = _LANCE_RESOLVERS.get(dataset_uri)
    if resolver is not None:
        return resolver

    with _LANCE_RESOLVER_LOCK:
        resolver = _LANCE_RESOLVERS.get(dataset_uri)
        if resolver is not None:
            return resolver

        id_column = os.getenv("COUNTIX_LANCE_VIDEO_ID_COLUMN", "video_id").strip() or "video_id"
        blob_column = os.getenv("COUNTIX_LANCE_VIDEO_BLOB_COLUMN", "video_blob").strip() or "video_blob"
        cache_dir = Path(os.path.expanduser(os.getenv("COUNTIX_LANCE_CACHE_DIR", "~/.cache/lmms_eval/countix_lance_videos")))

        resolver = LanceVideoBlobResolver(
            dataset_uri=dataset_uri,
            id_column=id_column,
            blob_column=blob_column,
            cache_dir=cache_dir,
            ext_column="video_ext",
            source_name="Countix Lance",
            video_extensions=_VIDEO_EXTENSIONS,
        )
        _LANCE_RESOLVERS[dataset_uri] = resolver
        return resolver


def _resolve_video_id(doc):
    for key in ("video_id", "clip_id", "id", "Video_Name", "video_name"):
        value = doc.get(key)
        if value:
            return Path(str(value).strip()).stem

    source_name = doc.get("source_name")
    if source_name:
        return Path(str(source_name).strip()).stem
    return ""


def _resolve_lance_video(doc):
    video_id = _resolve_video_id(doc)
    if not video_id:
        return None

    for dataset_uri in _candidate_lance_uris(doc):
        try:
            resolver = _get_lance_resolver(dataset_uri)
            return resolver.resolve(video_id)
        except FileNotFoundError:
            continue
        except Exception as exc:
            eval_logger.debug(f"Countix Lance resolve failed for uri={dataset_uri}, video_id={video_id}: {exc}")
            continue
    return None


def _extract_count(value):
    if value is None:
        return None
    text = str(value).strip().lower().replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    return int(round(float(match.group(0))))


def _get_target_count(doc):
    for key in ["count", "answer", "number", "gt_count", "label"]:
        target = _extract_count(doc.get(key))
        if target is not None:
            return target
    return None


def countix_doc_to_visual(doc):
    lance_resolved = _resolve_lance_video(doc)
    if lance_resolved is not None:
        return [lance_resolved]

    for key in ["video", "video_path", "media_path", "clip_path", "file", "path", "Video_Name", "video_name"]:
        value = doc.get(key)
        if value:
            return [resolve_media_reference(value, media_type="video", cache_dir="countix", env_vars=("COUNTIX_VIDEO_DIR",))]

    for key in ["clip_id", "video_id", "id"]:
        value = doc.get(key)
        if value:
            return [resolve_media_reference(str(value), media_type="video", cache_dir="countix", env_vars=("COUNTIX_VIDEO_DIR",))]
    return []


def countix_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    question = str(doc.get("question", "Count the number of repetitions in this clip.")).strip()
    return f"{pre_prompt}{question}{post_prompt}"


def countix_doc_to_target(doc):
    target = _get_target_count(doc)
    return "" if target is None else str(target)


def countix_process_results(doc, results):
    prediction = results[0] if results else ""
    pred_count = _extract_count(prediction)
    target_count = _get_target_count(doc)

    if pred_count is None or target_count is None:
        return {"mae_norm": 0.0 if target_count is None else float(abs(target_count)), "obo": 0.0}

    mae_norm = abs(pred_count - target_count) / (target_count + 0.1)
    obo = float(abs(pred_count - target_count) <= 1)
    return {"mae_norm": float(mae_norm), "obo": obo}
