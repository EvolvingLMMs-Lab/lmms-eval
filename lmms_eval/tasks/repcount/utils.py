import re

from lmms_eval.tasks._task_utils.media_resolver import resolve_media_reference

NUMBER_WORD_TO_NUMERAL = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}


def _extract_count(value):
    if value is None:
        return None
    normalized = str(value).strip().lower().replace(",", "")
    if normalized in NUMBER_WORD_TO_NUMERAL:
        return int(NUMBER_WORD_TO_NUMERAL[normalized])
    match = re.search(r"-?\d+(?:\.\d+)?", normalized)
    if not match:
        return None
    return int(round(float(match.group(0))))


def _get_target_count(doc):
    for key in ["count", "answer", "number", "gt_count", "label"]:
        count = _extract_count(doc.get(key))
        if count is not None:
            return count
    return None


def repcount_doc_to_visual(doc):
    for key in ["video", "video_path", "media_path", "clip_path", "file", "path"]:
        value = doc.get(key)
        if value:
            return [resolve_media_reference(value, media_type="video", cache_dir="repcount", env_vars=("REPCOUNT_VIDEO_DIR",))]

    for key in ["clip_id", "video_id", "id"]:
        value = doc.get(key)
        if value:
            return [resolve_media_reference(str(value), media_type="video", cache_dir="repcount", env_vars=("REPCOUNT_VIDEO_DIR",))]
    return []


def repcount_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    question = str(doc.get("question", "How many times is the action repeated in this video?")).strip()
    return f"{pre_prompt}{question}{post_prompt}"


def repcount_doc_to_target(doc):
    target = _get_target_count(doc)
    return "" if target is None else str(target)


def repcount_process_results(doc, results):
    prediction = results[0] if results else ""
    pred_count = _extract_count(prediction)
    target_count = _get_target_count(doc)

    if pred_count is None or target_count is None:
        return {"mae_norm": 0.0 if target_count is None else float(abs(target_count)), "obo": 0.0}

    mae_norm = abs(pred_count - target_count) / (target_count + 0.1)
    obo = float(abs(pred_count - target_count) <= 1)
    return {"mae_norm": float(mae_norm), "obo": obo}
