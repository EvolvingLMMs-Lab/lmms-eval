import re

from lmms_eval.tasks._task_utils.media_resolver import resolve_media_reference


def _normalize(text):
    lowered = str(text or "").strip().lower()
    lowered = re.sub(r"[^a-z0-9\s']", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def _word_error_rate(reference, hypothesis):
    ref_words = _normalize(reference).split()
    hyp_words = _normalize(hypothesis).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    rows = len(ref_words) + 1
    cols = len(hyp_words) + 1
    dp = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        dp[i][0] = i
    for j in range(cols):
        dp[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    return dp[-1][-1] / len(ref_words)


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


def av_asr_wer(items):
    if not items:
        return 0.0
    total = 0.0
    for item in items:
        total += _word_error_rate(item.get("gt", ""), item.get("pred", ""))
    return 100.0 * total / len(items)
