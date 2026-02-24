import re


def _normalize_text(text):
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _extract_predicted_label(prediction):
    return _normalize_text(prediction)


def ssv2_doc_to_visual(doc):
    for key in ["video", "video_path", "file", "path"]:
        value = doc.get(key)
        if value:
            return [value]
    return []


def ssv2_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    question = str(doc.get("question", "What action is being performed in this video?")).strip()
    return f"{pre_prompt}{question}{post_prompt}"


def ssv2_doc_to_target(doc):
    if "text" in doc:
        return str(doc["text"])
    if "label_text" in doc:
        return str(doc["label_text"])
    if "label" in doc:
        return str(doc["label"])
    return ""


def ssv2_process_results(doc, results):
    prediction = results[0] if results else ""
    pred = _extract_predicted_label(prediction)
    target = _normalize_text(ssv2_doc_to_target(doc))
    return {"acc": float(pred == target and target != "")}
