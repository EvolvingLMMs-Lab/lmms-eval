"""Omni-DuplexEval task utilities for lmms-eval.

Omni-DuplexEval (OpenBMB, https://github.com/OpenBMB/Omni-DuplexEval,
arXiv:2605.17360, data: Hothan/Omni-DuplexEval) evaluates *full-duplex*
omni-modal models: the model continuously watches a video stream after
hearing a spoken instruction and must (a) describe events in real time with
correct timing, or (b) proactively speak at the right moment (event reminders,
post-event reminders, corrections).

The model side is handled by a duplex streaming wrapper (e.g.
``minicpm_o_45_duplex``) which returns per-second chunks of
``{"text", "current_time", "is_listen", ...}``. This module ports the
*official* scoring logic from the Omni-DuplexEval repository:

- ``proactive_reminder.py`` — judge the model text inside a 10 s window after
  each gold reminder time (or the full response for corrections), returning
  0/1 success per event.
- ``real_time_description.py`` — per-sentence temporal sensitivity (0-3,
  multimodal judge over ffmpeg-extracted frames) and whole-response content
  accuracy (0.00-3.00, multimodal judge over the video).

Judge configuration (identical env vars to the official scripts):

    export DUPLEXEVAL_API_KEY=...      # or OPENAI_API_KEY
    export DUPLEXEVAL_BASE_URL=...     # or OPENAI_BASE_URL
    export DUPLEXEVAL_MODEL=...        # evaluator model id

The evaluator endpoint must be OpenAI-compatible; the content-accuracy judge
additionally needs video support (``{"type": "video"}`` message parts).
"""

import base64
import json
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from loguru import logger as eval_logger

WINDOW_SIZE = 10.0
FRAMES_PER_SECOND = 2
JUDGE_TEMPERATURE = 0.1


def _temporal_max_workers() -> int:
    """Per-rank judge concurrency; keep total (ranks x workers) within the
    judge endpoint's rate limit (OMNI_DUPLEXEVAL_JUDGE_WORKERS, default 2)."""
    try:
        return max(1, int(os.environ.get("OMNI_DUPLEXEVAL_JUDGE_WORKERS", "2")))
    except ValueError:
        return 2


TASK_ALIASES = {
    "correction": "correction",
    "pr_correction": "correction",
    "event_reminder": "proactive_reminder",
    "proactive_reminder": "proactive_reminder",
    "pr_event_reminder": "proactive_reminder",
    "post_event_reminder": "post_event_reminder",
    "pr_post_event_reminder": "post_event_reminder",
}


# ---------------------------------------------------------------------------
# Judge client (ported from Omni-DuplexEval llm_client.py)
# ---------------------------------------------------------------------------


class OpenAILLMClient:
    """Small wrapper around an OpenAI-compatible chat-completions endpoint."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, timeout: int = 120) -> None:
        import requests

        self.requests = requests
        self.api_key = api_key or os.environ.get("DUPLEXEVAL_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("DUPLEXEVAL_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        self.timeout = timeout

        if not self.api_key:
            raise ValueError("Missing API key. Set DUPLEXEVAL_API_KEY or OPENAI_API_KEY before running evaluation.")
        if not self.base_url:
            raise ValueError("Missing API base URL. Set DUPLEXEVAL_BASE_URL or OPENAI_BASE_URL before running evaluation.")

        self.base_url = self.base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "DuplexEval/1.0",
            "Accept": "application/json",
        }

    def chat_completion(self, messages, model: str, temperature: float = JUDGE_TEMPERATURE, max_tokens: Optional[int] = None) -> Dict[str, Any]:
        normalized = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            normalized.append({"role": role, "content": content})
        payload: Dict[str, Any] = {"model": model, "messages": normalized, "temperature": temperature}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        try:
            response = self.requests.post(f"{self.base_url}/chat/completions", headers=self.headers, json=payload, timeout=self.timeout)
            if response.status_code != 200:
                return {"error": response.text, "status_code": response.status_code}
            return response.json()
        except Exception as exc:
            return {"error": str(exc)}


def judge_model_id() -> str:
    return os.environ.get("DUPLEXEVAL_MODEL") or os.environ.get("OPENAI_MODEL_ID", "EVALUATOR_MODEL")


def extract_message_text(response: Dict[str, Any]) -> str:
    try:
        return str(response.get("choices", [{}])[0].get("message", {}).get("content", ""))
    except Exception:
        return ""


def _parse_json_block(text: str) -> Dict[str, Any]:
    for match in re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, flags=re.DOTALL):
        try:
            parsed = json.loads(re.sub(r",\s*([}\]])", r"\1", match))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return {}


def _parse_success_score(text: str) -> Dict[str, Any]:
    parsed = _parse_json_block(text)
    if "success_score" in parsed:
        parsed["success_score"] = int(float(parsed["success_score"]))
        return parsed
    score_match = re.search(r'"success_score"\s*:\s*([01])', text)
    return {"success_score": int(score_match.group(1)) if score_match else 0, "reasoning": ""}


def _parse_temporal_scores(text: str) -> Dict[str, Any]:
    parsed = _parse_json_block(text)
    if "temporal_score" in parsed:
        parsed["temporal_score"] = int(float(parsed["temporal_score"]))
    if "content_score" in parsed:
        parsed["content_score"] = round(float(parsed["content_score"]), 2)
    if "is_relevant" in parsed:
        parsed["is_relevant"] = int(float(parsed["is_relevant"]))
    if parsed:
        return parsed
    parsed = {}
    for pattern, key, caster in [
        (r'"temporal_score"\s*:\s*(\d+)', "temporal_score", int),
        (r'"is_relevant"\s*:\s*(\d+)', "is_relevant", int),
        (r'"content_score"\s*:\s*(\d+(?:\.\d+)?)', "content_score", float),
    ]:
        match = re.search(pattern, text)
        if match:
            parsed[key] = caster(match.group(1))
    return parsed


# ---------------------------------------------------------------------------
# Chunk normalization (ported from Omni-DuplexEval)
# ---------------------------------------------------------------------------


def _doc_video_duration(doc: dict) -> float:
    value = doc.get("video_duration")
    try:
        return float(value) if value else 0.0
    except (TypeError, ValueError):
        return 0.0


def normalize_chunks(raw: Any) -> List[Dict[str, Any]]:
    preamble = 0.0
    if isinstance(raw, str):
        raw = json.loads(raw) if raw.strip() else {}
    if isinstance(raw, dict):
        # duplex wrapper output: {"sessions": [chunk_log, ...], "preamble": s}
        # — the first session is the evaluated live-stream run, and
        # `preamble` is the user-speech audio prefilled before the live
        # stream. Subtract it so chunk times live on the media timeline
        # (gold reminder times and judge windows refer to the media).
        preamble = float(raw.get("preamble") or 0.0)
        sessions = raw.get("sessions")
        if isinstance(sessions, list) and sessions:
            raw = sessions[0]
        else:
            raw = raw.get("chunks", raw.get("sentences", []))
    if not isinstance(raw, list):
        return []
    chunks: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = item.get("text", item.get("sentence", item.get("content", "")))
        if not isinstance(text, str) or not text.strip():
            continue
        current_time = float(item.get("current_time", item.get("start", item.get("time", 0.0))) or 0.0)
        media_time = current_time - preamble
        # Chunks emitted while the spoken instruction was still streaming
        # predate the live media; mark them so media-timeline scoring (RTD
        # temporal/content) can exclude them, while correction tasks still
        # judge the full response (immediate corrections during the
        # instruction are legitimate).
        chunks.append(
            {
                "text": text.strip(),
                "current_time": max(0.0, media_time),
                "in_preamble": media_time < 0,
            }
        )
    return chunks


def chunks_to_sentences(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert duplex chunks into official {sentence, start, end} segments."""
    sentences = []
    for chunk in chunks:
        if chunk.get("in_preamble"):
            continue
        start = float(chunk.get("current_time", 0.0))
        sentences.append({"sentence": chunk.get("text", "").strip(), "start": start, "end": start})
    return [s for s in sentences if s["sentence"]]


def response_text_near_time(chunks: List[Dict[str, Any]], start_time: float, window_size: float) -> str:
    end_time = start_time + window_size
    selected = [chunk["text"] for chunk in chunks if start_time <= float(chunk.get("current_time", 0.0)) <= end_time and chunk.get("text")]
    return " ".join(selected).strip()


def full_response_text(chunks: List[Dict[str, Any]], include_preamble: bool = True) -> str:
    return " ".join(chunk.get("text", "") for chunk in chunks if chunk.get("text") and (include_preamble or not chunk.get("in_preamble"))).strip()


# ---------------------------------------------------------------------------
# Media materialization for parquet docs
# ---------------------------------------------------------------------------


def _materialize_dir() -> str:
    from lmms_eval.utils import resolve_cache_dir

    hf_home = os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface/"))
    path = resolve_cache_dir("omni_duplexeval_media", base_dir=hf_home)
    os.makedirs(path, exist_ok=True)
    return path


def _atomic_write(path: str, data: bytes) -> None:
    """Write bytes so readers never observe a partially written file."""
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "wb") as handle:
        handle.write(data)
    os.replace(tmp_path, path)


def materialize_media(value: Any, sample_id: str, field: str, suffix: str) -> str:
    """Return a local file path for a HF dataset media column value."""
    cache_path = os.path.join(_materialize_dir(), f"{field}_{sample_id}{suffix}")
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        return cache_path

    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        path = value.get("path")
        if path and os.path.exists(path):
            return str(path)
        bytes_value = value.get("bytes")
        if bytes_value:
            _atomic_write(cache_path, bytes(bytes_value))
            return cache_path
    path_attr = getattr(value, "path", None)
    if path_attr and os.path.exists(path_attr):
        return str(path_attr)
    if isinstance(value, (bytes, bytearray)):
        _atomic_write(cache_path, bytes(value))
        return cache_path
    raise ValueError(f"Could not materialize the {field!r} media field for sample {sample_id}.")


def _doc_sample_id(doc: dict) -> str:
    return str(doc.get("id", "")) or f"doc_{abs(hash(str(doc.get('question_text', '')))) % 10**8}"


# ---------------------------------------------------------------------------
# Doc accessors
# ---------------------------------------------------------------------------


def omni_duplexeval_doc_to_visual(doc: dict) -> List[str]:
    """Ordered duplex media: the spoken instruction first, then the live video.

    A duplex wrapper prefills audio files as the user's speech before
    streaming the video, which reproduces the Omni-DuplexEval protocol
    (instruction spoken by the user, model watches and speaks over time).
    """
    sample_id = _doc_sample_id(doc)
    visual = []
    question_audio = doc.get("question_audio")
    if question_audio is not None:
        visual.append(materialize_media(question_audio, sample_id, "question_audio", ".wav"))
    video = doc.get("video")
    if video is not None:
        visual.append(materialize_media(video, sample_id, "video", ".mp4"))
    return visual


def omni_duplexeval_doc_to_text(doc: dict) -> str:
    return doc.get("question_text", "")


# ---------------------------------------------------------------------------
# Proactive reminder / correction scoring (ported from proactive_reminder.py)
# ---------------------------------------------------------------------------


def build_reminder_prompt(instruction: str, response: str, task_type: str) -> str:
    if task_type == "post_event_reminder":
        timing_note = "The response is evaluated immediately after the target event has occurred."
    else:
        timing_note = "The response is evaluated at the target event time or immediately after it."

    return f"""
You are judging whether a model successfully completed an event-reminder task.

Task description:
The user gave an instruction asking the system to remind them when a specific event happens.
{timing_note}
The provided model text is the response segment generated in the evaluation window.

Input:
- User instruction: {instruction}
- Model response segment: "{response}"

Success criteria:
1. The response clearly refers to the target event in the instruction.
2. The response communicates a reminder, notification, or confirmation that the event has happened.
3. Vague narration, unrelated descriptions, or wrong-event reminders are failures.
4. The response does not need to match the instruction wording exactly if the meaning is clear.

Scoring:
- 1 = successful reminder.
- 0 = unsuccessful reminder.

Return exactly this JSON object:
{{
  "success_score": <0 or 1>,
  "reasoning": "<brief explanation>"
}}
"""


def build_correction_prompt(instruction: str, ground_answer: str, response: str) -> str:
    return f"""
You are judging whether a model successfully completed a correction task.

Task description:
The user instruction contains an incorrect statement about the video.
The model should identify the incorrect part and provide the correct information.

Input:
- User instruction: {instruction}
- Reference correction: {ground_answer}
- Model response: "{response}"

Success criteria:
1. Identify the error implied by the user instruction and the reference correction.
2. Check whether the model corrected all required error points.
3. The corrected content must preserve the correct context, including subject, object, action, and attributes.
4. Ignore extra information that is unrelated to both the instruction and the reference, unless it contradicts them.

Scoring:
- 1 = all required error points are corrected and the context is consistent.
- 0 = at least one required correction is missing, wrong, or contextually inconsistent.

Return exactly this JSON object:
{{
  "success_score": <0 or 1>,
  "reasoning": "<brief explanation covering the required correction points>"
}}
"""


def llm_judge_success(client: OpenAILLMClient, model_id: str, task_type: str, instruction: str, ground_answer: str, response: str) -> int:
    if task_type == "correction":
        prompt = build_correction_prompt(instruction, ground_answer, response)
    else:
        prompt = build_reminder_prompt(instruction, response, task_type)
    api_response = client.chat_completion(
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        model=model_id,
        max_tokens=1200,
    )
    if "error" in api_response:
        eval_logger.error(f"Omni-DuplexEval judge call failed: {str(api_response.get('error'))[:200]}")
        return 0
    return int(_parse_success_score(extract_message_text(api_response)).get("success_score", 0))


def _collect_reminder_times(doc: dict) -> List[float]:
    times = []
    for key in ("reminder1", "reminder2", "reminder_1", "reminder_2"):
        value = doc.get(key)
        if value is None:
            continue
        try:
            times.append(float(value))
        except (TypeError, ValueError):
            continue
    return times


def score_proactive_reminder(client: OpenAILLMClient, model_id: str, doc: dict, chunks: List[Dict[str, Any]], task_type: str) -> Dict[str, Any]:
    instruction = doc.get("question_text", "")
    reminders = _collect_reminder_times(doc)

    if task_type == "correction" or not reminders:
        response = full_response_text(chunks)
        score = llm_judge_success(client, model_id, task_type, instruction, str(doc.get("answer1", "")), response)
        return {"total_score": score, "event_scores": [score]}

    event_scores = []
    for start_time in reminders:
        response_segment = response_text_near_time(chunks, start_time, WINDOW_SIZE)
        if not response_segment:
            event_scores.append(0)
            continue
        event_scores.append(llm_judge_success(client, model_id, task_type, instruction, str(doc.get("answer1", "")), response_segment))
    return {"total_score": int(all(score == 1 for score in event_scores)), "event_scores": event_scores}


# ---------------------------------------------------------------------------
# Real-time description scoring (ported from real_time_description.py)
# ---------------------------------------------------------------------------


def get_video_duration(video_path: str) -> float:
    command = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprinting_wrappers=1:nokey=1", video_path]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as exc:
        eval_logger.warning(f"Failed to read video duration for {video_path}: {exc}")
        return 0.0


def extract_frames_for_time_range(video_path: str, start_time: float, end_time: float, frames_per_second: int = FRAMES_PER_SECOND) -> List[bytes]:
    if end_time <= start_time:
        return []
    segment_duration = end_time - start_time
    frame_count = max(1, int(segment_duration * frames_per_second))
    frames: List[bytes] = []
    with tempfile.TemporaryDirectory(prefix="omni_duplexeval_frames_") as temp_dir:
        for frame_index in range(frame_count):
            if frame_count == 1:
                frame_time = start_time + segment_duration / 2
            else:
                frame_time = start_time + frame_index * segment_duration / (frame_count - 1)
            frame_time = min(frame_time, max(start_time, end_time - 0.01))
            frame_path = os.path.join(temp_dir, f"frame_{frame_index:03d}.jpg")
            command = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                str(frame_time),
                "-i",
                video_path,
                "-vframes",
                "1",
                "-q:v",
                "2",
                "-y",
                frame_path,
            ]
            try:
                subprocess.run(command, capture_output=True, text=True, check=True)
                if os.path.exists(frame_path) and os.path.getsize(frame_path) > 0:
                    with open(frame_path, "rb") as handle:
                        frames.append(handle.read())
            except Exception as exc:
                eval_logger.warning(f"Frame extraction failed at {frame_time:.2f}s: {exc}")
    return frames


def generate_temporal_window(sentence_start: float, sentence_end: float, video_duration: float) -> Optional[tuple]:
    window_start = max(0.0, sentence_start - 2.0)
    window_end = max(window_start + 0.5, sentence_end - 2.0)
    if window_end <= video_duration and window_end - window_start >= 0.5:
        return window_start, window_end
    return None


def build_temporal_prompt(start_time: float, end_time: float, response_text: str, question: str) -> str:
    return f"""
You are evaluating a real-time video description system.

Basic information:
- Video segment: {start_time:.2f}s to {end_time:.2f}s.
- User instruction: {question}
- Model response sentence: "{response_text}"

Evaluation principles:
- Judge only the current video segment.
- Ignore response content that clearly belongs to earlier or later segments unless it contradicts the current segment.
- Language does not matter; evaluate semantic correctness and temporal alignment only.
- Decide whether the sentence is a substantive response or only a filler/polite phrase.

Temporal sensitivity score:
- 3: Excellent temporal alignment. The sentence accurately describes the current segment and follows the instruction.
- 2: Mostly aligned. Minor inaccuracies, omissions, or harmless references to nearby segments are allowed.
- 1: Poor alignment. Major inaccuracies, wrong timing, or mostly irrelevant content.
- 0: No meaningful alignment with the current segment.

Relevance label:
- 1: The sentence contains substantive task- or video-related information.
- 0: The sentence is only filler, acknowledgement, meta-commentary, or generic text without useful content.

Return exactly this JSON object:
{{
  "temporal_score": <0, 1, 2, or 3>,
  "temporal_reasoning": "<brief explanation>",
  "is_relevant": <0 or 1>
}}
"""


def _judge_temporal_sentence(sentence: Dict[str, Any], video_path: str, video_duration: float, question: str, model_id: str) -> Dict[str, Any]:
    client = OpenAILLMClient()
    window = generate_temporal_window(float(sentence["start"]), float(sentence["end"]), video_duration)
    if not window:
        return {"temporal_score": 0, "is_relevant": 0}
    window_start, window_end = window
    frame_bytes = extract_frames_for_time_range(video_path, window_start, window_end)
    if not frame_bytes:
        return {"temporal_score": 0, "is_relevant": 0}

    prompt = build_temporal_prompt(window_start, window_end, sentence["sentence"], question)
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for frame in frame_bytes:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(frame).decode('utf-8')}"}})
    # NOTE: no system message — some OpenAI-compatible reasoning endpoints
    # return empty content when a system message is combined with image parts.
    response = client.chat_completion(
        messages=[{"role": "user", "content": content}],
        model=model_id,
        max_tokens=3000,
    )
    if "error" in response:
        eval_logger.warning(f"Temporal judge call failed: {str(response.get('error'))[:150]}")
        return {"temporal_score": 0, "is_relevant": 0}
    parsed = _parse_temporal_scores(extract_message_text(response))
    return {"temporal_score": int(parsed.get("temporal_score", 0)), "is_relevant": int(parsed.get("is_relevant", 0))}


def build_content_prompt(response_text: str, question: str, references: List[str]) -> str:
    reference_block = ""
    if references:
        reference_block = "Reference annotations, for reference only:\n"
        for index, reference in enumerate(references, 1):
            reference_block += f"{index}. {reference}\n"

    return f"""
You are a precise content-accuracy evaluator for video description.

Basic information:
- User instruction: {question}
- Model response: "{response_text}"
{reference_block}
Evaluation goal:
Judge whether the model response is factually accurate for the whole video and aligned with the instruction.
Consider object/action/color/count/spatial/event errors, hallucinations, omissions, and irrelevant content.
Reference annotations are optional guidance; the primary evidence is the video itself.

Scoring:
- Use a decimal score from 0.00 to 3.00.
- Start from 3.00 and deduct for each error.
- Use 0.00 only when the response is empty, completely irrelevant, or contains no correct video facts.
- Output exactly two decimal places.

Return exactly this JSON object:
{{
  "content_score": <decimal from 0.00 to 3.00>,
  "content_reasoning": "<brief explanation with main errors and final score>"
}}
"""


CONTENT_FALLBACK_FRAMES = 8


def _extract_uniform_frames(video_path: str, num_frames: int) -> List[bytes]:
    """Extract uniformly-sampled JPEG frames for the content-judge fallback."""
    duration = get_video_duration(video_path)
    if duration <= 0:
        return []
    frames: List[bytes] = []
    with tempfile.TemporaryDirectory(prefix="omni_duplexeval_content_") as temp_dir:
        for index in range(num_frames):
            frame_time = duration * index / max(1, num_frames - 1)
            frame_path = os.path.join(temp_dir, f"frame_{index:03d}.jpg")
            command = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                str(frame_time),
                "-i",
                video_path,
                "-vframes",
                "1",
                "-q:v",
                "2",
                "-y",
                frame_path,
            ]
            try:
                subprocess.run(command, capture_output=True, text=True, check=True)
                if os.path.exists(frame_path) and os.path.getsize(frame_path) > 0:
                    with open(frame_path, "rb") as handle:
                        frames.append(handle.read())
            except Exception as exc:
                eval_logger.warning(f"Content fallback frame extraction failed at {frame_time:.2f}s: {exc}")
    return frames


def _judge_content_accuracy(client: OpenAILLMClient, video_path: str, response_text: str, question: str, references: List[str], model_id: str) -> float:
    prompt = build_content_prompt(response_text, question, references)
    response = client.chat_completion(
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "video", "video": video_path}]}],
        model=model_id,
        max_tokens=1200,
    )
    text_reply = extract_message_text(response) if "error" not in response else ""
    if text_reply.strip():
        parsed = _parse_temporal_scores(text_reply)
        if "content_score" in parsed:
            return round(float(parsed["content_score"]), 2)

    # Fallback: some OpenAI-compatible judge endpoints do not accept video
    # message parts; retry with uniformly-sampled frames as images.
    eval_logger.warning("Content judge returned no video verdict; retrying with sampled frames.")
    frames = _extract_uniform_frames(video_path, CONTENT_FALLBACK_FRAMES)
    if not frames:
        return 0.0
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt + "\n(The video is provided as uniformly-sampled frames in temporal order.)"}]
    for frame in frames:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(frame).decode('utf-8')}"}})
    response = client.chat_completion(messages=[{"role": "user", "content": content}], model=model_id, max_tokens=1200)
    if "error" in response:
        eval_logger.warning(f"Content judge fallback failed: {str(response.get('error'))[:150]}")
        return 0.0
    parsed = _parse_temporal_scores(extract_message_text(response))
    return round(float(parsed.get("content_score", 0.0)), 2)


def score_real_time_description(client: OpenAILLMClient, model_id: str, doc: dict, chunks: List[Dict[str, Any]]) -> Dict[str, float]:
    sample_id = _doc_sample_id(doc)
    video_path = materialize_media(doc.get("video"), sample_id, "video", ".mp4")
    question = doc.get("question_text", "")
    # The dataset ships an authoritative video_duration; prefer it over ffprobe
    # (some sandboxed runners ship an ffprobe that silently fails, which would
    # zero out every temporal window).
    video_duration = _doc_video_duration(doc) or get_video_duration(video_path)
    sentences = chunks_to_sentences(chunks)

    temporal_scores: List[int] = []
    with ThreadPoolExecutor(max_workers=_temporal_max_workers()) as executor:
        futures = [executor.submit(_judge_temporal_sentence, sentence, video_path, video_duration, question, model_id) for sentence in sentences]
        for future in as_completed(futures):
            result = future.result()
            if result.get("is_relevant", 0) == 1:
                temporal_scores.append(int(result["temporal_score"]))

    avg_temporal = sum(temporal_scores) / len(temporal_scores) if temporal_scores else 0.0
    references = [str(doc[key]).strip() for key in ("answer1", "answer2") if doc.get(key)]
    content_score = _judge_content_accuracy(client, video_path, full_response_text(chunks, include_preamble=False), question, references, model_id)
    return {"avg_temporal_score": round(avg_temporal, 4), "content_score": content_score}


# ---------------------------------------------------------------------------
# process_results entry points
# ---------------------------------------------------------------------------


def _client_once() -> OpenAILLMClient:
    try:
        return OpenAILLMClient()
    except ValueError as exc:
        eval_logger.error(f"Omni-DuplexEval judge not configured: {exc}")
        raise


def _pr_process(doc: dict, results: List[str], task_type: str) -> Dict[str, float]:
    chunks = normalize_chunks(results[0] if results else "[]")
    client = _client_once()
    outcome = score_proactive_reminder(client, judge_model_id(), doc, chunks, task_type)
    return {"omni_duplexeval_pr_success": float(outcome["total_score"])}


def omni_duplexeval_pr_event_reminder_process_results(doc: dict, results: List[str]) -> Dict[str, float]:
    return _pr_process(doc, results, "proactive_reminder")


def omni_duplexeval_pr_post_event_reminder_process_results(doc: dict, results: List[str]) -> Dict[str, float]:
    return _pr_process(doc, results, "post_event_reminder")


def omni_duplexeval_pr_correction_process_results(doc: dict, results: List[str]) -> Dict[str, float]:
    return _pr_process(doc, results, "correction")


def omni_duplexeval_rtd_process_results(doc: dict, results: List[str]) -> Dict[str, float]:
    chunks = normalize_chunks(results[0] if results else "[]")
    client = _client_once()
    scores = score_real_time_description(client, judge_model_id(), doc, chunks)
    return {
        "omni_duplexeval_rtd_temporal": float(scores["avg_temporal_score"]),
        "omni_duplexeval_rtd_content": float(scores["content_score"]),
    }


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------


def _mean_aggregate(results: List[float]) -> float:
    values = [float(value) for value in results if value is not None]
    return sum(values) / len(values) if values else 0.0


def omni_duplexeval_pr_aggregate(results: List[float]) -> float:
    return _mean_aggregate(results)


def omni_duplexeval_rtd_temporal_aggregate(results: List[float]) -> float:
    return _mean_aggregate(results)


def omni_duplexeval_rtd_content_aggregate(results: List[float]) -> float:
    return _mean_aggregate(results)
