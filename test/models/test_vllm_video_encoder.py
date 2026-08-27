"""Encoder-selection contract for the chat vLLM wrapper's video path.

Qwen3-VL-family models read `<t seconds>` markers to perceive a frame clip as
temporal input; other models take bare frames (issue #1404). The wrapper must
route video encoding through ``to_qwen3_vl_openai_messages`` only when
``is_qwen3_vl`` is set, on both request paths.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from lmms_eval.models.chat.vllm import VLLM
from lmms_eval.models.simple import vllm as simple_vllm
from lmms_eval.protocol import ChatMessages


def _video_model(is_qwen3_vl: bool) -> VLLM:
    model = VLLM.__new__(VLLM)
    model.task_dict = {"task": {"test": [{"question": "What happens in the video?"}]}}
    model.max_new_tokens = 16
    model.max_pixels = 1605632
    model.min_image_pixels = 28
    model.max_frame_num = 32
    model.fps = None
    model.nframes = 32
    model.is_qwen3_vl = is_qwen3_vl
    return model


def _request():
    def doc_to_messages(doc):
        return [
            {
                "role": "user",
                "content": [{"type": "text", "text": doc["question"]}],
            }
        ]

    return SimpleNamespace(arguments=("context", doc_to_messages, {}, 0, "task", "test"))


@pytest.fixture
def encoder_calls(monkeypatch) -> list[str]:
    calls: list[str] = []

    def fake_openai(self, video_kwargs=None, **kwargs):
        calls.append("openai")
        return [{"role": "user", "content": [{"type": "text", "text": "bare"}]}]

    def fake_qwen3(self, video_kwargs=None):
        calls.append("qwen3vl")
        return [{"role": "user", "content": [{"type": "text", "text": "<0.0 seconds>"}]}]

    monkeypatch.setattr(ChatMessages, "to_openai_messages", fake_openai)
    monkeypatch.setattr(ChatMessages, "to_qwen3_vl_openai_messages", fake_qwen3)
    return calls


def test_make_one_request_routes_video_by_flag(encoder_calls):
    model = _video_model(True)
    model.make_one_request(_request())
    assert encoder_calls == ["qwen3vl"]

    encoder_calls.clear()
    model = _video_model(False)
    model.make_one_request(_request())
    assert encoder_calls == ["openai"]


def test_multi_round_path_routes_video_by_flag(encoder_calls):
    model = _video_model(True)
    messages = model._to_openai_messages([{"role": "user", "content": [{"type": "text", "text": "hello"}]}])
    assert encoder_calls == ["qwen3vl"]
    assert messages[0]["content"][0]["text"] == "<0.0 seconds>"

    model = _video_model(False)
    model._to_openai_messages([{"role": "user", "content": [{"type": "text", "text": "hello"}]}])
    assert encoder_calls == ["qwen3vl", "openai"]


def test_simple_vllm_uses_shared_video_decoder(monkeypatch):
    model = simple_vllm.VLLM.__new__(simple_vllm.VLLM)
    model.max_frame_num = 3
    model.video_decode_backend = "torchcodec"
    model.min_image_pixels = 1
    model._enforce_image_resize = False
    observed = {}

    def fake_read_video(video_path, **kwargs):
        observed.update({"video_path": video_path, **kwargs})
        return np.stack([np.full((2, 2, 3), value, dtype=np.uint8) for value in range(3)])

    monkeypatch.setattr(simple_vllm, "read_video", fake_read_video)
    monkeypatch.setattr(simple_vllm, "encode_image_to_base64", lambda image, **kwargs: int(np.asarray(image)[0, 0, 0]))

    encoded = model.encode_video("demo.mp4")

    assert encoded == [0, 1, 2]
    assert observed == {
        "video_path": "demo.mp4",
        "num_frm": 3,
        "force_include_last_frame": True,
        "backend": "torchcodec",
    }
