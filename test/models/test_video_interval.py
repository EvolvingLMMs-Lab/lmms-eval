"""A requested video interval must survive the chat message and the Gemini part."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import av
import numpy as np

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
