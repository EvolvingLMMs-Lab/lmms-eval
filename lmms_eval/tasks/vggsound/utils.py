import re


def _normalize_label(text):
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def vggsound_doc_to_audio(doc):
    for key in ["audio", "audio_path", "file", "path"]:
        value = doc.get(key)
        if value:
            return [value]
    return []


def vggsound_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    question = str(doc.get("question", "What is the main sound event in this clip?")).strip()
    return f"{pre_prompt}{question}{post_prompt}"


def vggsound_doc_to_target(doc):
    for key in ["label", "answer", "class", "sound_class"]:
        value = doc.get(key)
        if value is not None:
            return str(value)
    return ""


def vggsound_process_results(doc, results):
    prediction = _normalize_label(results[0] if results else "")
    target = _normalize_label(vggsound_doc_to_target(doc))
    return {"acc": float(prediction == target and target != "")}
