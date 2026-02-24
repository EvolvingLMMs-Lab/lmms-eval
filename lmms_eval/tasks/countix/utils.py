import re


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
    for key in ["video", "video_path", "image", "img", "file", "path"]:
        value = doc.get(key)
        if value:
            return [value]
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
