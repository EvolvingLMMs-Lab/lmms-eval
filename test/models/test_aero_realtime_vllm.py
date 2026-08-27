from types import SimpleNamespace

import numpy as np

from lmms_eval.models.chat.aero_realtime_vllm import (
    RealtimeSample,
    _frame_time,
    _iter_realtime_chunks,
    _normalize_video_content,
    _truncate_at_until,
)


def test_normalize_video_content_supports_protocol_range():
    content = SimpleNamespace(url="/tmp/video.mp4", start_time=3.0, end_time=7.5)
    assert _normalize_video_content(content) == ("/tmp/video.mp4", 3.0, 7.5)


def test_normalize_video_content_supports_legacy_range_dict():
    content = SimpleNamespace(
        url={"url": "/tmp/video.mp4", "video_start": 3.0, "video_end": 7.5},
        start_time=None,
        end_time=None,
    )
    assert _normalize_video_content(content) == ("/tmp/video.mp4", 3.0, 7.5)


def test_truncate_at_until_uses_earliest_stop():
    result = _truncate_at_until("answer<end>trailing###more", ["###", "<end>"])
    assert result == "answer"


def test_frame_time_is_relative_to_selected_clip():
    metadata = {"frames_indices": [600, 610], "fps": 10.0}
    assert _frame_time(metadata, 0, 1.0) == 0.0
    assert _frame_time(metadata, 1, 1.0) == 1.0


def test_fallback_question_chunk_precedes_decode_tail():
    sample = RealtimeSample(
        audio=np.ones(8, dtype=np.float32),
        video=np.zeros((0, 1, 1, 3), dtype=np.uint8),
        video_metadata={},
        sample_fps=1.0,
    )
    chunks = list(
        _iter_realtime_chunks(
            sample,
            audio_chunk_ms=1.0,
            ask_text="question",
            ask_second=1.0,
            extra_silence_chunks=1,
        )
    )
    question_chunk, decode_chunk = chunks[-2:]
    assert question_chunk["text"] == "question"
    assert decode_chunk["timestamp"] > question_chunk["timestamp"]
