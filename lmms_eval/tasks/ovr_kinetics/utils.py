import re

from lmms_eval.tasks._task_utils.media_resolver import resolve_media_reference


def _extract_count(value):
    if value is None:
        return None
    text = str(value).strip().lower().replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    return int(round(float(match.group(0))))


def _get_target_count(doc):
    for key in ["count", "answer", "gt_count", "label"]:
        target = _extract_count(doc.get(key))
        if target is not None:
            return target
    return None


def ovr_kinetics_doc_to_visual(doc):
    for key in ["video", "video_path", "media_path", "clip_path", "file", "path"]:
        value = doc.get(key)
        if value:
            return [resolve_media_reference(value, media_type="video", cache_dir="ovr_kinetics", env_vars=("OVR_KINETICS_VIDEO_DIR",))]

    for key in ["clip_id", "video_id", "id"]:
        value = doc.get(key)
        if value:
            return [resolve_media_reference(str(value), media_type="video", cache_dir="ovr_kinetics", env_vars=("OVR_KINETICS_VIDEO_DIR",))]
    return []


def ovr_kinetics_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    text_description = str(doc.get("text_description", doc.get("text", "the described action"))).strip()
    question = str(doc.get("question", f"How many times does '{text_description}' happen in this video?")).strip()
    return f"{pre_prompt}{question}{post_prompt}"


def ovr_kinetics_doc_to_target(doc):
    target = _get_target_count(doc)
    return "" if target is None else str(target)


def ovr_kinetics_process_results(doc, results):
    prediction = results[0] if results else ""
    pred_count = _extract_count(prediction)
    target_count = _get_target_count(doc)

    if pred_count is None or target_count is None:
        return {"mae": 0.0 if target_count is None else float(abs(target_count)), "obo": 0.0}

    mae = abs(pred_count - target_count)
    obo = float(abs(pred_count - target_count) <= 1)
    return {"mae": float(mae), "obo": obo}
