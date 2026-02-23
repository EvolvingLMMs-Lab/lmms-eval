import re
from typing import Any

from PIL import Image

_COUNT_KEYS = ["annotated_pos_count", "pos_count", "count", "answer", "label", "gt_count", "gt_num"]
_CAPTION_KEYS = ["pos_caption", "caption", "question", "query", "prompt", "text"]
_IMAGE_KEYS = ["image", "img", "query_image"]


def _to_rgb(image_obj: Any):
    if isinstance(image_obj, Image.Image):
        return image_obj.convert("RGB")
    return None


def _to_int(value: Any):
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        return int(round(float(value)))

    text = str(value).strip().replace(",", "")
    if not text:
        return None

    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None

    try:
        return int(round(float(match.group(0))))
    except ValueError:
        return None


def _extract_count(doc: dict):
    for key in _COUNT_KEYS:
        count = _to_int(doc.get(key))
        if count is not None:
            return count
    return None


def _extract_caption(doc: dict) -> str:
    for key in _CAPTION_KEYS:
        value = doc.get(key)
        if isinstance(value, str) and value.strip():
            text = value.strip()
            text = re.sub(r"^[Tt]he\s+", "", text)
            return text.rstrip(". ")
    return "objects"


def fsc147_doc_to_visual(doc):
    visuals = []
    for key in _IMAGE_KEYS:
        image_obj = _to_rgb(doc.get(key))
        if image_obj is not None:
            visuals.append(image_obj)
            break
    return visuals


def fsc147_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")

    object_phrase = _extract_caption(doc)
    question = f"How many {object_phrase} are there in the image?"
    return f"{pre_prompt}{question}{post_prompt}"


def fsc147_doc_to_target(doc):
    target = _extract_count(doc)
    if target is None:
        return ""
    return str(target)


def fsc147_process_results(doc, results):
    prediction = str(results[0]).strip() if results else ""
    target_count = _extract_count(doc)

    if target_count is None:
        return {"fsc147_exact_match": 0.0, "fsc147_mae": 0.0}

    pred_count = _to_int(prediction)
    if pred_count is None:
        return {"fsc147_exact_match": 0.0, "fsc147_mae": float(abs(target_count))}

    return {
        "fsc147_exact_match": float(pred_count == target_count),
        "fsc147_mae": float(abs(pred_count - target_count)),
    }


def fsc147_aggregate_exact_match(items):
    if not items:
        return 0.0
    return sum(float(item) for item in items) / len(items)


def fsc147_aggregate_mae(items):
    if not items:
        return 0.0
    return sum(float(item) for item in items) / len(items)
