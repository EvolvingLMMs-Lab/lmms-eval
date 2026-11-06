"""Qwen2.5-VL chat and simple backends: sampled fps reaches the processor.

CPU only, no model weights: a real `Qwen2_5_VLProcessor` with a tiny local
tokenizer and synthetic frames checks that `second_per_grid_ts` follows the
frame budget.
"""

import json

import numpy as np
import pytest
from transformers import Qwen2_5_VLProcessor, Qwen2TokenizerFast, Qwen2VLImageProcessor, Qwen2VLVideoProcessor
from transformers.video_utils import VideoMetadata

from lmms_eval.models.chat.qwen2_5_vl import _sampled_fps
from lmms_eval.models.simple import qwen2_5_vl as simple_qwen

SOURCE_FRAMES = 300
SOURCE_FPS = 30
FRAME_BUDGETS = (8, 16)


def _expected_sampled_fps(nframes):
    return nframes / SOURCE_FRAMES * SOURCE_FPS


def test_sampled_fps_from_dict_frame_indices():
    md = {"fps": SOURCE_FPS, "total_num_frames": SOURCE_FRAMES, "frames_indices": list(range(8))}
    assert _sampled_fps(md) == pytest.approx(0.8)


def test_sampled_fps_explicit_value_and_fallbacks():
    assert _sampled_fps({"sampled_fps": 0.8}) == 0.8
    assert _sampled_fps({"fps": 30}) == 30.0
    assert _sampled_fps({}) == 2.0


@pytest.mark.parametrize("nframes", FRAME_BUDGETS)
def test_sampled_fps_from_video_metadata_object(nframes):
    md = VideoMetadata(total_num_frames=SOURCE_FRAMES, fps=SOURCE_FPS, frames_indices=list(range(nframes)))
    assert _sampled_fps(md) == pytest.approx(_expected_sampled_fps(nframes))


@pytest.fixture
def processor(tmp_path):
    vocab = {"<|endoftext|>": 0, "<|image_pad|>": 1, "<|video_pad|>": 2}
    (tmp_path / "vocab.json").write_text(json.dumps(vocab))
    (tmp_path / "merges.txt").write_text("#version: 0.2\n")
    tokenizer = Qwen2TokenizerFast(
        vocab_file=str(tmp_path / "vocab.json"),
        merges_file=str(tmp_path / "merges.txt"),
        additional_special_tokens=["<|image_pad|>", "<|video_pad|>"],
    )
    return Qwen2_5_VLProcessor(image_processor=Qwen2VLImageProcessor(), tokenizer=tokenizer, video_processor=Qwen2VLVideoProcessor())


def _second_per_grid_ts(processor, nframes, metadata):
    out = processor(
        text=["<|video_pad|>"],
        videos=[np.zeros((nframes, 56, 56, 3), dtype=np.uint8)],
        video_metadata=[metadata],
        videos_kwargs={"do_sample_frames": False, "fps": _sampled_fps(metadata)},
        return_tensors="pt",
    )
    return out["second_per_grid_ts"].tolist()


@pytest.mark.parametrize(
    "make_metadata",
    [
        lambda n: {"fps": SOURCE_FPS, "total_num_frames": SOURCE_FRAMES, "frames_indices": list(range(n))},
        lambda n: VideoMetadata(total_num_frames=SOURCE_FRAMES, fps=SOURCE_FPS, frames_indices=list(range(n))),
    ],
    ids=["dict", "VideoMetadata"],
)
def test_second_per_grid_ts_follows_frame_budget(processor, make_metadata):
    values = {n: _second_per_grid_ts(processor, n, make_metadata(n)) for n in FRAME_BUDGETS}
    for n, ts in values.items():
        # temporal patch size 2: one grid spans 2 sampled frames
        assert ts == pytest.approx([2 / _expected_sampled_fps(n)])
    assert values[8] != values[16]


def _fake_process_vision_info(total_frames):
    """Stand-in for qwen-vl-utils: one video of `total_frames` frames sampled at 2 fps."""

    def fake(messages, return_video_kwargs=False, return_video_metadata=False):
        frames = np.zeros((total_frames, 56, 56, 3), dtype=np.uint8)
        md = {"fps": SOURCE_FPS, "total_num_frames": SOURCE_FRAMES, "frames_indices": list(range(total_frames))}
        return None, [(frames, md)], {"do_sample_frames": False}

    return fake


@pytest.mark.parametrize("nframes", FRAME_BUDGETS)
def test_simple_path_sampled_fps_follows_frame_cap(monkeypatch, processor, nframes):
    monkeypatch.setattr(simple_qwen, "process_vision_info", _fake_process_vision_info(32))
    _, videos, processor_kwargs = simple_qwen._process_vision_inputs([], nframes)
    assert videos[0].shape[0] == nframes
    # 32 of 300 source frames were sampled; the cap keeps nframes of them
    assert processor_kwargs["videos_kwargs"]["fps"] == pytest.approx(_expected_sampled_fps(nframes))
    out = processor(text=["<|video_pad|>"], videos=videos, return_tensors="pt", **processor_kwargs)
    assert out["second_per_grid_ts"].tolist() == pytest.approx([2 / _expected_sampled_fps(nframes)])


def test_simple_path_without_videos_passes_no_kwargs(monkeypatch):
    monkeypatch.setattr(simple_qwen, "process_vision_info", lambda *a, **k: ("img", None, {}))
    assert simple_qwen._process_vision_inputs([], 8) == ("img", None, {})


def test_limit_video_inputs_without_metadata_keeps_old_return():
    videos = [np.zeros((20, 8, 8, 3), dtype=np.uint8)]
    assert simple_qwen._limit_video_inputs(videos, 8)[0].shape[0] == 8
