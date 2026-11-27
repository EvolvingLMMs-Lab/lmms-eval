import re

import editdistance

from lmms_eval.tasks._task_utils.media_resolver import resolve_media_reference


def _normalize(text):
    lowered = str(text or "").strip().lower()
    lowered = re.sub(r"[^a-z0-9\s']", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def av_asr_doc_to_visual(doc):
    visuals = []
    for key in ["audio", "audio_path"]:
        value = doc.get(key)
        if value:
            visuals.append(resolve_media_reference(value, media_type="audio", cache_dir="av_asr", env_vars=("AV_ASR_AUDIO_DIR", "AV_ASR_MEDIA_DIR")))
            break
    for key in ["video", "video_path", "file", "path"]:
        value = doc.get(key)
        if value:
            visuals.append(resolve_media_reference(value, media_type="video", cache_dir="av_asr", env_vars=("AV_ASR_VIDEO_DIR", "AV_ASR_MEDIA_DIR")))
            break

    if not visuals:
        for key in ["clip_id", "id", "sample_id"]:
            value = doc.get(key)
            if value:
                clip_value = str(value)
                visuals.append(resolve_media_reference(clip_value, media_type="audio", cache_dir="av_asr", env_vars=("AV_ASR_AUDIO_DIR", "AV_ASR_MEDIA_DIR")))
                visuals.append(resolve_media_reference(clip_value, media_type="video", cache_dir="av_asr", env_vars=("AV_ASR_VIDEO_DIR", "AV_ASR_MEDIA_DIR")))
                break

    visuals = [item for item in visuals if item]
    return visuals


def av_asr_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    question = str(doc.get("question", "Transcribe the speech in this video.")).strip()
    return f"{pre_prompt}{question}{post_prompt}"


def av_asr_doc_to_target(doc):
    for key in ["text", "transcript", "gt", "answer"]:
        value = doc.get(key)
        if value is not None:
            return str(value)
    return ""


def av_asr_process_results(doc, results):
    prediction = results[0] if results else ""
    target = av_asr_doc_to_target(doc)
    return {"wer": {"gt": target, "pred": prediction}}


def av_asr_wer(items: list[dict[str, str]]) -> float:
    """Return corpus WER as a percentage, aligning each utterance separately."""
    errors = reference_words = 0
    for item in items:
        reference = _normalize(item.get("gt", "")).split()
        hypothesis = _normalize(item.get("pred", "")).split()
        errors += editdistance.eval(reference, hypothesis)
        reference_words += len(reference)
    # With no reference words, each insertion counts as one error.
    return 100.0 * errors / (reference_words or 1)
