"""
Unit tests for two bugs in LlavaHf.generate_until (lmms_eval/models/simple/llava_hf.py):

Bug A: UnboundLocalError when task_type == "text" (e.g. ScienceQA samples with no image).
  `image_tokens` was only assigned inside if/elif branches for image/video but referenced
  unconditionally on the next line, crashing on text-only samples.

Bug B: TypeError from batch_decode when model.generate raises an exception.
  The except handler set cont = "" (an empty string); batch_decode expects a tensor or
  list-of-id-lists, so passing "" caused a Rust-binding TypeError.
"""

from __future__ import annotations

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

# decord is not available on all platforms; stub it before importing llava_hf
_decord_stub = types.ModuleType("decord")
_decord_stub.VideoReader = MagicMock()
_decord_stub.cpu = MagicMock()
sys.modules.setdefault("decord", _decord_stub)

from lmms_eval.models.simple.llava_hf import LlavaHf  # noqa: E402


def _make_request(context: str, doc_to_visual, doc_id: int = 0, task: str = "demo", split: str = "test") -> SimpleNamespace:
    return SimpleNamespace(args=(context, {}, doc_to_visual, doc_id, task, split))


def _make_llava_hf(model_generate_side_effect=None) -> LlavaHf:
    """Return a minimal LlavaHf instance that bypasses __init__."""
    instance = LlavaHf.__new__(LlavaHf)

    import torch

    tokenizer = MagicMock()
    tokenizer.chat_template = None
    tokenizer.eos_token_id = 2  # eot_token_id is a property returning tokenizer.eos_token_id
    tokenizer.encode.return_value = [1, 2, 3]
    tokenizer.decode.return_value = "</s>"
    tokenizer.apply_chat_template.return_value = "USER: test\nASSISTANT:"
    tokenizer.batch_decode.return_value = ["mock output"]

    model = MagicMock()
    model.dtype = torch.float32
    if model_generate_side_effect is not None:
        model.generate.side_effect = model_generate_side_effect
    else:
        model.generate.return_value = torch.zeros((1, 5), dtype=torch.long)

    image_processor = MagicMock()
    fake_inputs = MagicMock()
    fake_inputs.__getitem__ = lambda self, key: torch.zeros((1, 3), dtype=torch.long) if key == "input_ids" else MagicMock()
    fake_inputs.to.return_value = fake_inputs
    image_processor.return_value = fake_inputs

    cache_hook = MagicMock()
    accelerator = MagicMock()
    accelerator.is_main_process = False

    # Set via backing attributes because tokenizer/model/device/rank are @property
    instance._tokenizer = tokenizer
    instance._model = model
    instance._image_processor = image_processor
    instance._device = "cpu"
    instance._rank = 0
    instance._world_size = 1
    instance.use_cache = True
    instance.chat_template = None
    instance.batch_size_per_gpu = 1
    instance.max_frames_num = 32
    instance.accelerator = accelerator
    instance.cache_hook = cache_hook
    instance.task_dict = {"demo": {"test": [{"id": 0}]}}
    return instance


class TestLlavaHfGenerateUntil(unittest.TestCase):

    def test_text_task_no_unbound_error(self):
        """Bug A: text-only sample (empty visuals) must not raise UnboundLocalError."""
        instance = _make_llava_hf()
        # doc_to_visual returns [] → task_type == "text"
        request = _make_request("What is the answer?", doc_to_visual=lambda _doc: [])

        # Before the fix this raised: UnboundLocalError: local variable 'image_tokens' referenced before assignment
        try:
            results = instance.generate_until([request])
        except UnboundLocalError as e:
            self.fail(f"generate_until raised UnboundLocalError for text task: {e}")

        self.assertEqual(len(results), 1)

    def test_generate_failure_returns_empty_string(self):
        """Bug B: when model.generate raises, batch_decode must not crash with TypeError."""
        import PIL.Image

        instance = _make_llava_hf(model_generate_side_effect=RuntimeError("OOM"))
        # doc_to_visual returns a real PIL image → task_type == "image", enters generate
        dummy_image = PIL.Image.new("RGB", (64, 64))
        request = _make_request("Describe the image.", doc_to_visual=lambda _doc: [dummy_image])

        # Before the fix: TypeError: argument 'ids': Can't extract 'str' to 'Vec'
        try:
            results = instance.generate_until([request])
        except TypeError as e:
            self.fail(f"generate_until raised TypeError after generate failure: {e}")

        self.assertEqual(len(results), 1)
        # Result must be a string (empty or otherwise), not an exception
        self.assertIsInstance(results[0], str)


if __name__ == "__main__":
    unittest.main()
