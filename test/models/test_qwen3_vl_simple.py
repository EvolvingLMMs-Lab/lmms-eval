import builtins
import importlib
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from lmms_eval.models.simple.qwen3_vl import Qwen3_VL, _is_video_path


class _FakeTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def encode(self, text):
        return [1, 2, 3]

    def decode(self, token_id):
        return "<eos>"


class _FakeInputs(dict):
    def __init__(self):
        super().__init__(input_ids=torch.tensor([[10, 11]]))

    @property
    def input_ids(self):
        return self["input_ids"]

    def to(self, device):
        return self


class _FakeProcessor:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        return ["prompt"]

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeInputs()

    def batch_decode(self, generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return ["final answer"]


class _FakeModel:
    def generate(self, **kwargs):
        return torch.tensor([[10, 11, 12]])


class _VideoMetadata:
    def __init__(self, frames_indices):
        self.frames_indices = np.asarray(frames_indices)


class TestQwen3VLSimple(unittest.TestCase):
    def setUp(self):
        # Fail before loading the native library, even when Decord is installed.
        original_import = builtins.__import__
        original_import_module = importlib.import_module

        def reject_decord(importer):
            def guarded(name, *args, **kwargs):
                if name == "decord" or name.startswith("decord."):
                    raise AssertionError("Unexpected Decord import")
                return importer(name, *args, **kwargs)

            return guarded

        for target, importer in [("builtins.__import__", original_import), ("importlib.import_module", original_import_module)]:
            patcher = patch(target, reject_decord(importer))
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_is_video_path_supports_common_video_extensions(self):
        for path in ("clip.mp4", "clip.avi", "clip.mov", "clip.mkv", "clip.webm", "clip.mpeg", "clip.mpg", "clip.MPEG"):
            with self.subTest(path=path):
                self.assertTrue(_is_video_path(path))

        self.assertFalse(_is_video_path("frame.jpg"))
        self.assertFalse(_is_video_path(None))

    def _make_model(self, max_num_frames=3):
        model = Qwen3_VL.__new__(Qwen3_VL)
        model._tokenizer = _FakeTokenizer()
        model.processor = _FakeProcessor()
        model._model = _FakeModel()
        model.max_pixels = 1024
        model.min_pixels = 256
        model.max_num_frames = max_num_frames
        model.fps = None
        model.total_pixels = None
        model.enable_thinking = None
        model.system_prompt = "You are a helpful assistant."
        model.interleave_visuals = False
        model.reasoning_prompt = None
        model.batch_size_per_gpu = 1
        model.use_cache = False
        model.device_map = "cpu"
        model._device = torch.device("cpu")
        model._rank = 0
        model._world_size = 1
        model.task_dict = {"demo_task": {"test": [{"id": 0}]}}
        model.cache_hook = types.SimpleNamespace(add_partial=lambda *args, **kwargs: None)
        return model

    def test_generate_until_passes_video_metadata_and_kwargs_to_processor(self):
        model = self._make_model(max_num_frames=3)
        metadata = _VideoMetadata([0, 10, 20, 30, 40])
        video_tensor = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        request = types.SimpleNamespace(
            args=("Describe the video", {}, lambda doc: ["demo.mp4"], 0, "demo_task", "test"),
        )

        with (
            patch("lmms_eval.models.simple.qwen3_vl.process_vision_info", return_value=(None, [(video_tensor, metadata)], {"fps": 30.0, "max_frames": 5})) as process_vision,
            patch("lmms_eval.models.simple.qwen3_vl._probe_video_metadata", return_value=(2, 30.0)) as probe,
        ):
            result = model.generate_until([request])

        self.assertEqual(result, ["final answer"])
        probe.assert_called_once_with("demo.mp4", count_frames=True)
        video_message = process_vision.call_args.args[0][0][1]["content"][0]
        self.assertEqual(video_message["nframes"], 2)
        self.assertEqual(len(model.processor.calls), 1)

        processor_call = model.processor.calls[0]
        self.assertIs(processor_call["videos"][0], video_tensor)
        self.assertIs(processor_call["video_metadata"][0], metadata)
        self.assertEqual(processor_call["fps"], 30.0)
        self.assertEqual(processor_call["max_frames"], 5)
        self.assertTrue(np.array_equal(metadata.frames_indices, np.array([0, 10, 20, 30, 40])))

    def test_image_and_text_requests_do_not_probe_video(self):
        for visuals in (None, [], [Image.new("RGB", (2, 2))]):
            with self.subTest(visuals=visuals):
                model = self._make_model()
                request = types.SimpleNamespace(args=("Describe", {}, lambda doc: visuals, 0, "demo_task", "test"))
                with (
                    patch("lmms_eval.models.simple.qwen3_vl.process_vision_info", return_value=(visuals, None, {})),
                    patch("lmms_eval.models.simple.qwen3_vl._probe_video_metadata") as probe,
                ):
                    self.assertEqual(model.generate_until([request]), ["final answer"])
                probe.assert_not_called()

    def test_fps_and_total_pixels_video_modes_do_not_probe(self):
        for mode, value in (("fps", 1), ("total_pixels", 1024)):
            with self.subTest(mode=mode):
                model = self._make_model()
                setattr(model, mode, value)
                request = types.SimpleNamespace(args=("Describe", {}, lambda doc: ["demo.mp4"], 0, "demo_task", "test"))
                with (
                    patch("lmms_eval.models.simple.qwen3_vl.process_vision_info", return_value=(None, None, {})) as process_vision,
                    patch("lmms_eval.models.simple.qwen3_vl._probe_video_metadata") as probe,
                ):
                    self.assertEqual(model.generate_until([request]), ["final answer"])
                probe.assert_not_called()
                video_message = process_vision.call_args.args[0][0][1]["content"][0]
                self.assertNotIn("nframes", video_message)
                self.assertEqual(video_message[mode], value)


if __name__ == "__main__":
    unittest.main()
