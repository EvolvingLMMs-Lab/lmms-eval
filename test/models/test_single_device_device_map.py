"""LlavaHf on a single process must not pass the empty default device_map to from_pretrained.

transformers turns a string device_map into torch.device(device_map), and torch.device("") raises, so the default
device_map="" failed on a single process unless device_map was set explicitly.
"""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# decord is not available on all platforms; stub it before importing llava_hf
_decord_stub = types.ModuleType("decord")
_decord_stub.VideoReader = MagicMock()
_decord_stub.cpu = MagicMock()
sys.modules.setdefault("decord", _decord_stub)

import lmms_eval.models.simple.llava_hf as llava_hf_module  # noqa: E402


@pytest.mark.parametrize(("device_map", "expected"), [("", "cpu"), ("cpu", "cpu"), ("auto", "auto")])
def test_llava_hf_single_process_device_map(device_map, expected):
    model_cls = MagicMock()
    with (
        patch.object(llava_hf_module.AutoConfig, "from_pretrained", return_value=MagicMock(model_type="llava")),
        patch.dict(llava_hf_module.model_map, {"llava": model_cls}),
        patch.object(llava_hf_module.AutoProcessor, "from_pretrained", return_value=MagicMock()),
    ):
        llava_hf_module.LlavaHf(pretrained="dummy", device="cpu", device_map=device_map)

    assert model_cls.from_pretrained.call_args.kwargs["device_map"] == expected
