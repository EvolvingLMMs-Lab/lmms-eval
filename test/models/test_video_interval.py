"""A requested video interval must survive the chat message, the Gemini part and the audio track."""

from __future__ import annotations

import ast
import math
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import pytest

from lmms_eval.models.model_utils.video_interval import resolve_video_bounds, uniform_indices
from lmms_eval.protocol import ChatMessages

COLORS = (
    ("red", (255, 0, 0)),
    ("green", (0, 255, 0)),
    ("blue", (0, 0, 255)),
    ("yellow", (255, 255, 0)),
    ("magenta", (255, 0, 255)),
    ("cyan", (0, 255, 255)),
)
ROOT = Path(__file__).resolve().parents[2]


def _source(path: Path, name: str) -> str:
    text = path.read_text()
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(text, node)
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return ast.get_source_segment(text, item)
    raise LookupError(name)


class _Task:
    _IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
    _AUDIO_EXTS = {".wav", ".mp3"}


class _Part:
    kind = "uri"
    video_metadata = None

    @staticmethod
    def from_uri(file_uri, mime_type):
        part = _Part()
        part.file_uri = file_uri
        part.mime_type = mime_type
        return part

    @staticmethod
    def from_text(text):
        return {"kind": "text", "text": text}

    @staticmethod
    def from_bytes(data, mime_type):
        return {"kind": "bytes", "mime_type": mime_type}


class _VideoMetadata:
    def __init__(self, start_offset=None, end_offset=None):
        self.start_offset = start_offset
        self.end_offset = end_offset


class _Types:
    Part = _Part
    VideoMetadata = _VideoMetadata

    class Content:
        def __init__(self, role, parts):
            self.role = role
            self.parts = parts


def _load_visual_to_content():
    from PIL import Image as PIL_Image

    namespace = {"os": __import__("os"), "PIL_Image": PIL_Image}
    exec(_source(ROOT / "lmms_eval/api/task.py", "_visual_to_content"), namespace)
    return namespace["_visual_to_content"]


def _load_gemini():
    namespace = {"types": _Types, "Optional": __import__("typing").Optional}
    simple = ROOT / "lmms_eval/models/simple/gemini.py"
    for name in ("_format_video_offset", "gemini_video_part"):
        exec(_source(simple, name), namespace)
    chat = ROOT / "lmms_eval/models/chat/gemini.py"
    exec(_source(chat, "_chat_messages_to_gemini_contents"), namespace)
    return namespace["_chat_messages_to_gemini_contents"]


def _write_video(path: Path) -> None:
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=1)
    stream.width = 32
    stream.height = 32
    stream.pix_fmt = "yuv420p"
    for _name, color in COLORS:
        array = np.empty((32, 32, 3), dtype=np.uint8)
        array[:, :] = color
        frame = av.VideoFrame.from_ndarray(array, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


RATE = 8000


def _write_av_video(path: Path, channels: int = 1, rate: int = RATE, codec: str = "pcm_s16le", ext: str = "mkv", audio_from_second: int = 0) -> None:
    """Six-color 1 fps video muxed with one non-silent PCM tone per second and channel.

    Second ``i`` of channel ``c`` carries a ``(200 + i * 50 + c * 100)`` Hz sine at
    amplitude 10000, so audio timing and channel placement can be verified
    sample-exactly against the source, not just by presence. ``audio_from_second``
    mutes the track's early seconds (the stream exists from there) to model a late
    audio track whose real A/V offset must survive the window.
    """
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=1)
    stream.width = 32
    stream.height = 32
    stream.pix_fmt = "yuv420p"
    audio = container.add_stream(codec, rate=rate)
    audio.layout = "stereo" if channels == 2 else "mono"
    for index, (_name, color) in enumerate(COLORS):
        array = np.empty((32, 32, 3), dtype=np.uint8)
        array[:, :] = color
        frame = av.VideoFrame.from_ndarray(array, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
        if index < audio_from_second:
            continue
        waves = [np.round(10000 * np.sin(np.arange(rate) * 2 * np.pi * (200 + index * 50 + channel * 100) / rate)).astype(np.int16) for channel in range(channels)]
        samples = np.stack(waves, axis=1).reshape(1, -1)  # packed s16 interleaves per time step: L0, R0, L1, R1...
        tone = av.AudioFrame.from_ndarray(samples, format="s16", layout=audio.layout)
        tone.sample_rate = rate
        tone.pts = index * rate
        tone.time_base = Fraction(1, rate)
        for packet in audio.encode(tone):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    for packet in audio.encode():
        container.mux(packet)
    container.close()


def _audio(path: Path) -> tuple[np.ndarray, int]:
    """Decoded audio as packed s16 at the stream's own rate and layout."""
    with av.open(str(path)) as container:
        astream = container.streams.audio[0]
        rate = int(astream.rate)
        resampler = av.AudioResampler(format="s16", layout=astream.layout, rate=astream.rate)
        chunks = [piece.to_ndarray() for frame in container.decode(audio=0) for piece in resampler.resample(frame)]
        chunks.extend(piece.to_ndarray() for piece in resampler.resample(None) if chunks)
    return (np.concatenate(chunks, axis=1) if chunks else np.zeros((1, 0), np.int16)), rate


def _colors(path: Path, start: float | None, end: float | None) -> list[str]:
    container = av.open(str(path))
    stream = container.streams.video[0]
    fps = float(stream.average_rate)
    frames = []
    for index, frame in enumerate(container.decode(video=0)):
        pixel = tuple(int(value) for value in frame.to_ndarray(format="rgb24")[16, 16])
        name = min(COLORS, key=lambda item: sum((left - right) ** 2 for left, right in zip(item[1], pixel)))[0]
        frames.append((index, name))
    container.close()
    if start is None and end is None:
        return [name for _index, name in frames]
    start_frame = math.ceil(start * fps) if start is not None else 0
    end_frame = math.floor(end * fps) if end is not None else len(frames) - 1
    return [name for index, name in frames if start_frame <= index <= end_frame]


def test_video_interval_reaches_hf_and_gemini(tmp_path):
    video_path = tmp_path / "clip.mp4"
    _write_video(video_path)
    visual_to_content = _load_visual_to_content()
    convert = _load_gemini()
    content = visual_to_content(
        _Task,
        {"type": "video", "path": str(video_path), "video_start": 2, "video_end": 4},
    )
    messages = ChatMessages(messages=[{"role": "user", "content": [content, {"type": "text", "text": "color?"}]}])
    video = messages.messages[0].content[0]
    hf_video = messages.to_hf_messages()[0]["content"][0]

    assert video.url == str(video_path)
    assert video.start_time == 2.0
    assert video.end_time == 4.0
    assert hf_video["video"] == str(video_path)
    assert hf_video["video_start"] == 2.0
    assert hf_video["video_end"] == 4.0

    uploaded = []

    def upload(path):
        uploaded.append(path)
        return type("File", (), {"uri": path, "mime_type": "video/mp4"})()

    contents = convert(messages, upload, [])
    part = contents[0].parts[0]
    assert uploaded == [str(video_path)]
    assert part.video_metadata.start_offset == "2s"
    assert part.video_metadata.end_offset == "4s"
    assert _colors(video_path, 2, 4) == ["blue", "yellow", "magenta"]


def test_zero_start_is_kept():
    visual_to_content = _load_visual_to_content()
    content = visual_to_content(_Task, {"type": "video", "path": "clip.mp4", "video_start": 0, "video_end": 1})
    messages = ChatMessages(messages=[{"role": "user", "content": [content]}])
    hf_video = messages.to_hf_messages()[0]["content"][0]
    assert messages.messages[0].content[0].start_time == 0.0
    assert hf_video["video_start"] == 0.0
    assert hf_video["video_end"] == 1.0


def test_unbounded_video_has_no_gemini_clip(tmp_path):
    video_path = tmp_path / "whole.mp4"
    _write_video(video_path)
    visual_to_content = _load_visual_to_content()
    convert = _load_gemini()
    content = visual_to_content(_Task, str(video_path))
    messages = ChatMessages(messages=[{"role": "user", "content": [content, {"type": "text", "text": "whole"}]}])

    def upload(path):
        return type("File", (), {"uri": path, "mime_type": "video/mp4"})()

    part = convert(messages, upload, [])[0].parts[0]
    assert messages.messages[0].content[0].start_time is None
    assert part.video_metadata is None
    assert _colors(video_path, None, None) == [name for name, _color in COLORS]


def test_legacy_nested_dict_still_clips():
    convert = _load_gemini()
    messages = ChatMessages(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "video", "url": {"path": "clip.mp4", "video_start": 2, "video_end": 4}},
                    {"type": "text", "text": "color?"},
                ],
            }
        ]
    )

    def upload(path):
        return type("File", (), {"uri": path, "mime_type": "video/mp4"})()

    part = convert(messages, upload, [])[0].parts[0]
    assert part.file_uri == "clip.mp4"
    assert part.video_metadata.start_offset == "2s"
    assert part.video_metadata.end_offset == "4s"


def test_phi4_samples_inside_the_requested_interval():
    indexes = uniform_indices(6, 1.0, 3, 2, 4).tolist()
    assert indexes == [2, 3, 4]
    assert uniform_indices(6, 1.0, 6).tolist() == [0, 1, 2, 3, 4, 5]
    path, start, end = resolve_video_bounds({"path": "clip.mp4", "video_start": 0, "video_end": 1})
    assert path == "clip.mp4"
    assert start == 0
    assert end == 1
    assert uniform_indices(6, 1.0, 2, start, end).tolist()[0] == 0


def test_sync_huggingface_processor_receives_only_requested_frames(tmp_path: Path) -> None:
    from lmms_eval.models.chat.huggingface import Huggingface

    path = tmp_path / "source.mp4"
    _write_video(path)
    bounds = [(2, 4), (None, None), (0, 0), (1.25, 3), (None, 1), (5, None)]
    parts = [{"type": "video", "url": str(path), "start_time": start, "end_time": end} for start, end in bounds]
    received: list[str] = []

    class Captured(Exception):
        pass

    class InspectProcessor:
        def apply_chat_template(self, *args, **kwargs) -> str:
            return "probe"

        def __call__(self, *, videos: list[str], **kwargs) -> None:
            received.extend(videos)
            assert videos[1] == str(path)  # normal processor input, even next to bounded clips
            colors = [_colors(Path(video), None, None) for video in videos]
            assert colors == [["blue", "yellow", "magenta"], [name for name, _ in COLORS], ["red"], ["blue", "yellow"], ["red", "green"], ["cyan"]]
            raise Captured

    model = object.__new__(Huggingface)
    model.batch_size_per_gpu, model._rank, model.system_prompt = 1, 1, None
    model.processor = InspectProcessor()
    model.task_dict = {"probe": {"test": [[{"role": "user", "content": parts}]]}}
    req = SimpleNamespace(args=("probe", lambda doc: doc, {}, 0, "probe", "test"))
    with pytest.raises(Captured):
        model.generate_until([req])
    assert all(not Path(video).exists() for video in received if video != str(path))
    assert path.exists()


@pytest.mark.parametrize(("start", "end"), [(2.2, 2.8), (6, None), (4, 2)])
def test_sync_huggingface_empty_interval_is_not_the_full_video(tmp_path: Path, start: float, end: float | None) -> None:
    from contextlib import ExitStack

    from lmms_eval.models.chat.huggingface import _processor_video

    path = tmp_path / "source.mp4"
    _write_video(path)
    with ExitStack() as stack, pytest.raises(ValueError, match="interval"):
        _processor_video({"video": str(path), "video_start": start, "video_end": end}, stack)


def test_sync_huggingface_bounded_clips_keep_the_audio_track(tmp_path: Path) -> None:
    """Bounded requests must hand the processor frames AND the audio they present.

    Each case pins the decoded clip samples to the exact source slice, so audio
    content, offset and duration are all verified, not just the stream count.
    """
    from lmms_eval.models.chat.huggingface import Huggingface

    path = tmp_path / "source.mkv"
    _write_av_video(path)
    source, _rate = _audio(path)
    assert source.shape == (1, 6 * RATE) and int(np.max(np.abs(source))) == 10000  # non-silent control
    cases = [
        ((2, 4), slice(2 * RATE, 4 * RATE), ["blue", "yellow", "magenta"]),  # audio is [2s, 4s): 16000 samples
        ((0, None), slice(0, 6 * RATE), [name for name, _ in COLORS]),  # zero start keeps everything to the end
        ((0, 0), None, ["red"]),  # point window: video only, [0, 0) contains no audio
        ((None, 1), slice(0, RATE), ["red", "green"]),
        ((5, None), slice(5 * RATE, 6 * RATE), ["cyan"]),
        ((None, None), None, [name for name, _ in COLORS]),  # no interval: the declared file itself
    ]
    parts = [{"type": "video", "url": str(path), "start_time": start, "end_time": end} for (start, end), _slice, _names in cases]

    class Captured(Exception):
        pass

    class InspectProcessor:
        def apply_chat_template(self, *args, **kwargs) -> str:
            return "probe"

        def __call__(self, *, videos: list[str], **kwargs) -> None:
            # Clips live only inside generate_until's ExitStack, so verify them here.
            assert len(videos) == len(cases)
            for (bounds, window, names), video in zip(cases, videos):
                if bounds == (None, None):
                    assert video == str(path)  # pass-through: audio trivially untouched
                    continue
                clip = Path(video)
                assert clip != path and clip.exists()
                with av.open(str(clip)) as container:
                    streams = [s.type for s in container.streams]
                assert _colors(clip, None, None) == names
                if window is None:  # the [0, 0) point window carries no audio at all
                    assert streams == ["video"] and bounds == (0, 0)
                    continue
                assert streams == ["video", "audio"]
                audio, clip_rate = _audio(clip)
                assert clip_rate == RATE
                assert np.array_equal(audio, source[:, window])  # sample-exact content, offset and duration
            raise Captured

    model = object.__new__(Huggingface)
    model.batch_size_per_gpu, model._rank, model.system_prompt = 1, 1, None
    model.processor = InspectProcessor()
    model.task_dict = {"probe": {"test": [[{"role": "user", "content": parts}]]}}
    req = SimpleNamespace(args=("probe", lambda doc: doc, {}, 0, "probe", "test"))
    with pytest.raises(Captured):
        model.generate_until([req])
    assert path.exists()


def test_bounded_clip_of_a_video_only_source_has_no_audio_stream(tmp_path: Path) -> None:
    """A source without audio yields a video-only clip; absence is never fabricated."""
    from contextlib import ExitStack

    from lmms_eval.models.chat.huggingface import _processor_video

    path = tmp_path / "video-only.mp4"
    _write_video(path)
    with ExitStack() as stack:
        clip = _processor_video({"video": str(path), "video_start": 2, "video_end": 4}, stack)
        with av.open(clip) as container:
            assert len(container.streams.video) == 1 and len(container.streams.audio) == 0
        assert _colors(Path(clip), None, None) == ["blue", "yellow", "magenta"]


def test_sync_huggingface_stereo_clips_keep_both_channels(tmp_path: Path) -> None:
    """Stereo bounded clips keep both channels sample-exact on one time base.

    Guards the mux ordering regression: every output stream must exist before the
    first packet is muxed, or the Matroska muxer aborts the process natively
    (add_stream after the container header corrupts the trailer). This test failed
    with a native abort when the audio stream was created late.
    """
    from lmms_eval.models.chat.huggingface import Huggingface

    rate = 48000
    path = tmp_path / "stereo.mkv"
    _write_av_video(path, channels=2, rate=rate)
    source, _rate = _audio(path)
    assert source.shape == (1, 6 * rate * 2) and int(np.max(np.abs(source))) == 10000
    cases = [
        ((2, 4), slice(2 * rate * 2, 4 * rate * 2), ["blue", "yellow", "magenta"]),  # [2s, 4s): 96000 samples/channel
        ((0, 0), None, ["red"]),  # point window: video only
        ((5, None), slice(5 * rate * 2, 6 * rate * 2), ["cyan"]),
        ((None, None), None, [name for name, _ in COLORS]),
    ]
    parts = [{"type": "video", "url": str(path), "start_time": start, "end_time": end} for (start, end), _slice, _names in cases]

    class Captured(Exception):
        pass

    class InspectProcessor:
        def apply_chat_template(self, *args, **kwargs) -> str:
            return "probe"

        def __call__(self, *, videos: list[str], **kwargs) -> None:
            assert len(videos) == len(cases)
            for (bounds, window, names), video in zip(cases, videos):
                if bounds == (None, None):
                    assert video == str(path)
                    continue
                clip = Path(video)
                assert clip != path and clip.exists()
                with av.open(str(clip)) as container:
                    streams = [s.type for s in container.streams]
                assert _colors(clip, None, None) == names
                if window is None:  # the [0, 0) point window carries no audio at all
                    assert streams == ["video"] and bounds == (0, 0)
                    continue
                assert streams == ["video", "audio"]
                audio, clip_rate = _audio(clip)
                assert clip_rate == rate
                assert np.array_equal(audio, source[:, window])  # both channels, sample-exact
            raise Captured

    model = object.__new__(Huggingface)
    model.batch_size_per_gpu, model._rank, model.system_prompt = 1, 1, None
    model.processor = InspectProcessor()
    model.task_dict = {"probe": {"test": [[{"role": "user", "content": parts}]]}}
    req = SimpleNamespace(args=("probe", lambda doc: doc, {}, 0, "probe", "test"))
    with pytest.raises(Captured):
        model.generate_until([req])
    assert path.exists()


def test_sync_huggingface_stereo_aac_mp4_clip_keeps_audio(tmp_path: Path) -> None:
    """The planar AAC decode path (fltp -> s16) stays sample-exact through the clip."""
    from lmms_eval.models.chat.huggingface import Huggingface

    rate = 44100
    path = tmp_path / "stereo-aac.mp4"
    _write_av_video(path, channels=2, rate=rate, codec="aac", ext="mp4")
    source, _rate = _audio(path)
    # AAC overshoots peaks and appends encoder padding, so assert non-silence here and
    # keep the sample-exact comparison to the window slices, which sit in the clean region.
    assert source.shape[0] == 1 and int(np.max(np.abs(source))) > 5000 and source.shape[1] > 6 * rate * 2
    cases = [((2, 4), 2, 4, ["blue", "yellow", "magenta"]), ((0, 0), None, None, ["red"])]
    parts = [{"type": "video", "url": str(path), "start_time": start, "end_time": end} for (start, end), _a, _b, _names in cases]

    class Captured(Exception):
        pass

    class InspectProcessor:
        def apply_chat_template(self, *args, **kwargs) -> str:
            return "probe"

        def __call__(self, *, videos: list[str], **kwargs) -> None:
            assert len(videos) == len(cases)
            for ((start, end), span_start, span_end, names), video in zip(cases, videos):
                clip = Path(video)
                assert clip.exists()
                with av.open(str(clip)) as container:
                    streams = [s.type for s in container.streams]
                assert _colors(clip, None, None) == names
                if span_start is None:  # the [0, 0) point window carries no audio at all
                    assert streams == ["video"] and (start, end) == (0, 0)
                    continue
                assert streams == ["video", "audio"]
                audio, clip_rate = _audio(clip)
                assert clip_rate == rate
                expected = source[:, span_start * rate * 2 : span_end * rate * 2]
                assert audio.shape == expected.shape
                assert np.array_equal(audio, expected)  # deterministic AAC decode + same s16 conversion
            raise Captured

    model = object.__new__(Huggingface)
    model.batch_size_per_gpu, model._rank, model.system_prompt = 1, 1, None
    model.processor = InspectProcessor()
    model.task_dict = {"probe": {"test": [[{"role": "user", "content": parts}]]}}
    req = SimpleNamespace(args=("probe", lambda doc: doc, {}, 0, "probe", "test"))
    with pytest.raises(Captured):
        model.generate_until([req])
    assert path.exists()


def test_sync_huggingface_late_audio_track_keeps_its_real_offset(tmp_path: Path) -> None:
    """A late audio track must keep its A/V offset on the common media axis.

    The track only carries sound from second 3, so for [2, 4) the clip's audio starts
    one second into the clip (a pts gap, not silence padding, and not content pulled
    from outside the window); re-anchoring the stream to its own first sample would
    shift this content to the clip's start and lose the offset.
    """
    from lmms_eval.models.chat.huggingface import Huggingface

    path = tmp_path / "late-audio.mkv"
    _write_av_video(path, audio_from_second=3)
    source, _rate = _audio(path)
    # The decoder yields the sparse track's content contiguously (seconds 3, 4, 5), so
    # media second 3 is the FIRST decoded second; its media position is asserted via pts.
    assert source.shape == (1, 3 * RATE)
    content_second_3 = slice(0, RATE)
    cases = [((2, 4), 1.0, content_second_3), ((None, 4), 3.0, content_second_3)]
    parts = [{"type": "video", "url": str(path), "start_time": start, "end_time": end} for (start, end), _p, _s in cases]

    class Captured(Exception):
        pass

    class InspectProcessor:
        def apply_chat_template(self, *args, **kwargs) -> str:
            return "probe"

        def __call__(self, *, videos: list[str], **kwargs) -> None:
            assert len(videos) == len(cases)
            for ((start, end), expected_pts_s, window), video in zip(cases, videos):
                clip = Path(video)
                assert clip.exists()
                with av.open(str(clip)) as container:
                    stream = container.streams.audio[0]
                    time_base = stream.time_base
                    frames = [(f.pts * time_base, f.to_ndarray()) for f in container.decode(audio=0)]
                expected_frames = {"(2, 4)": ["blue", "yellow", "magenta"], "(None, 4)": [name for name, _ in COLORS][:5]}[str((start, end))]
                assert _colors(clip, None, None) == expected_frames  # frame pts inclusive of the end bound
                assert len(frames) == 1
                assert abs(frames[0][0] - expected_pts_s) < 1e-9  # the real offset survives
                assert np.array_equal(frames[0][1], source[:, window])  # content from its own media seconds
            raise Captured

    model = object.__new__(Huggingface)
    model.batch_size_per_gpu, model._rank, model.system_prompt = 1, 1, None
    model.processor = InspectProcessor()
    model.task_dict = {"probe": {"test": [[{"role": "user", "content": parts}]]}}
    req = SimpleNamespace(args=("probe", lambda doc: doc, {}, 0, "probe", "test"))
    with pytest.raises(Captured):
        model.generate_until([req])
    assert path.exists()


def test_sync_huggingface_fractional_window_cuts_samples_not_seconds(tmp_path: Path) -> None:
    """A window that no frame boundary aligns with trims audio at sample precision."""
    from lmms_eval.models.chat.huggingface import Huggingface

    path = tmp_path / "fractional.mkv"
    _write_av_video(path)
    source, _rate = _audio(path)
    parts = [{"type": "video", "url": str(path), "start_time": 2.5, "end_time": 3.75}]

    class Captured(Exception):
        pass

    class InspectProcessor:
        def apply_chat_template(self, *args, **kwargs) -> str:
            return "probe"

        def __call__(self, *, videos: list[str], **kwargs) -> None:
            clip = Path(videos[0])
            assert clip.exists()
            assert _colors(clip, None, None) == ["yellow"]  # frames with pts in [2.5, 3.75]
            with av.open(str(clip)) as container:
                assert [s.type for s in container.streams] == ["video", "audio"]
                frame = next(container.decode(video=0))
                assert float(frame.pts * frame.time_base) == pytest.approx(0.5)
            with av.open(str(clip)) as container:
                sample = next(container.decode(audio=0))
                assert float(sample.pts * sample.time_base) == pytest.approx(0.0)
            audio, clip_rate = _audio(clip)
            assert clip_rate == RATE
            expected = source[:, int(2.5 * RATE) : int(3.75 * RATE)]  # 10000 samples, not a whole second
            assert np.array_equal(audio, expected)
            raise Captured

    model = object.__new__(Huggingface)
    model.batch_size_per_gpu, model._rank, model.system_prompt = 1, 1, None
    model.processor = InspectProcessor()
    model.task_dict = {"probe": {"test": [[{"role": "user", "content": parts}]]}}
    req = SimpleNamespace(args=("probe", lambda doc: doc, {}, 0, "probe", "test"))
    with pytest.raises(Captured):
        model.generate_until([req])
    assert path.exists()
