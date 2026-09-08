"""Guard Decord-free adapters without loading optional model dependencies."""

import ast
from pathlib import Path

import numpy as np
import pytest

MODELS = Path(__file__).resolve().parents[2] / "lmms_eval" / "models"


@pytest.mark.parametrize(
    "adapter",
    [
        "simple/qwen2_vl.py",
        "simple/qwen2_5_vl_interleave.py",
        "simple/qwen3_vl.py",
        "simple/llava_onevision1_5.py",
        "chat/qwen2_5_vl.py",
    ],
)
def test_adapters_leave_video_decoding_to_the_selected_backend(adapter):
    tree = ast.parse((MODELS / adapter).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            assert "decord" not in node.id.lower()
        elif isinstance(node, ast.alias):
            assert "decord" not in node.name.lower()
        elif isinstance(node, ast.ImportFrom):
            assert "decord" not in (node.module or "").lower()
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert node.value != "decord"


def test_llava_vid_zero_frames_does_not_import_decord():
    # Isolate this method so the test does not require the external llava package.
    path = MODELS / "simple" / "llava_vid.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LlavaVid")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "load_video")

    def reject_decord():
        raise AssertionError("Decord import requested")

    namespace = {"np": np, "import_decord": reject_decord}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)

    frames = namespace["load_video"](None, "unused.mp4", 0, 1)
    assert frames.shape == (1, 336, 336, 3)
    assert not frames.any()
    with pytest.raises(AssertionError, match="Decord import requested"):
        namespace["load_video"](None, "unused.mp4", 1, 1)
