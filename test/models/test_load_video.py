import importlib

import numpy as np
import pytest

from lmms_eval.models.model_utils import load_video


class _FakeDecordBatch:
    def __init__(self, frames):
        self._frames = frames

    def asnumpy(self):
        return self._frames


class _FakeVideoReader:
    last_indices = None

    def __init__(self, path, ctx, num_threads):
        self.path = path
        self.ctx = ctx
        self.num_threads = num_threads
        self.frames = np.stack([np.full((2, 2, 3), value, dtype=np.uint8) for value in range(10)])

    def __len__(self):
        return len(self.frames)

    def get_avg_fps(self):
        return 10.0

    def get_batch(self, indices):
        type(self).last_indices = indices
        return _FakeDecordBatch(self.frames[indices])


class _FakeDecord:
    VideoReader = _FakeVideoReader

    @staticmethod
    def cpu(device_id):
        return ("cpu", device_id)


def test_shared_decord_backend_is_lazy_and_uses_canonical_indices(monkeypatch):
    real_import_module = importlib.import_module

    def fake_import_module(name, *args, **kwargs):
        if name == "decord":
            return _FakeDecord
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(load_video.importlib, "import_module", fake_import_module)

    frames = load_video.read_video("demo.mp4", num_frm=4, backend="decord", force_include_last_frame=True)

    assert _FakeVideoReader.last_indices == [0, 3, 6, 9]
    assert frames[:, 0, 0, 0].tolist() == [0, 3, 6, 9]


def test_legacy_decord_helper_keeps_duplicate_short_video_indices(monkeypatch):
    real_import_module = importlib.import_module

    def fake_import_module(name, *args, **kwargs):
        if name == "decord":
            return _FakeDecord
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(load_video.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(
        _FakeVideoReader,
        "__init__",
        lambda self, path, ctx, num_threads: setattr(
            self,
            "frames",
            np.stack([np.full((2, 2, 3), value, dtype=np.uint8) for value in range(3)]),
        ),
    )

    frames = load_video.load_video_decord("short.mp4", max_frames_num=5)

    assert _FakeVideoReader.last_indices == [0, 0, 1, 1, 2]
    assert frames[:, 0, 0, 0].tolist() == [0, 0, 1, 1, 2]


def _write_matroska_video(path, frame_count):
    av = pytest.importorskip("av")
    container = av.open(str(path), mode="w")
    stream = container.add_stream("ffv1", rate=10)
    stream.width = 16
    stream.height = 16
    stream.pix_fmt = "yuv420p"

    for index in range(frame_count):
        array = np.full((16, 16, 3), index * 10, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(array, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def test_pyav_packet_fallback_samples_instead_of_returning_every_frame(tmp_path):
    video_path = tmp_path / "unknown-frame-count.mkv"
    _write_matroska_video(video_path, frame_count=12)

    frames = load_video.read_video(video_path.as_posix(), num_frm=3, backend="pyav", force_include_last_frame=True)

    assert frames.shape == (3, 16, 16, 3)
    assert np.allclose(frames.mean(axis=(1, 2, 3)), [0, 50, 110], atol=2)


def test_pyav_short_video_returns_each_source_frame_once(tmp_path):
    video_path = tmp_path / "short.mkv"
    _write_matroska_video(video_path, frame_count=4)

    frames = load_video.read_video(video_path.as_posix(), num_frm=8, backend="pyav", force_include_last_frame=True)

    assert frames.shape == (4, 16, 16, 3)
    assert np.allclose(frames.mean(axis=(1, 2, 3)), [0, 10, 20, 30], atol=2)
