import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from lmms_eval.models.chat.qwen2_5_vl import Qwen2_5_VL


class _FakeTokenizer:
    eos_token_id = 0
    pad_token_id = 0


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

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return ["prompt"]

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeInputs()

    def batch_decode(self, generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return ["final answer"]


class _FakeModel:
    def generate(self, **kwargs):
        return torch.tensor([[10, 11, 12]])


class _FakeVideoReader:
    def __init__(self, path):
        self.path = path

    def __len__(self):
        return 5


class _FakeVideoMetadata:
    def __init__(self, frames_indices, fps=10.0, total_num_frames=5):
        self.frames_indices = np.asarray(frames_indices)
        self.fps = fps
        self.total_num_frames = total_num_frames

    @property
    def sampled_fps(self):
        return len(self.frames_indices) / self.total_num_frames * self.fps


class _FakeChatMessages:
    last_video_kwargs = None

    def __init__(self, messages):
        self.messages = messages

    def extract_media(self):
        return [], ["demo.mp4"], []

    def to_hf_messages(self, video_kwargs=None):
        type(self).last_video_kwargs = dict(video_kwargs or {})
        return [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": "demo.mp4", **(video_kwargs or {})},
                    {"type": "text", "text": "Describe the video"},
                ],
            }
        ]


class TestQwen25VLChat(unittest.TestCase):
    def _make_model(self, max_num_frames=3, fps=None):
        model = Qwen2_5_VL.__new__(Qwen2_5_VL)
        model._tokenizer = _FakeTokenizer()
        model.processor = _FakeProcessor()
        model._model = _FakeModel()
        model.max_pixels = 1024
        model.min_pixels = 256
        model.max_num_frames = max_num_frames
        model.fps = fps
        model.batch_size_per_gpu = 1
        model.use_cache = False
        model.device_map = "cpu"
        model._device = torch.device("cpu")
        model._rank = 0
        model._world_size = 1
        model.task_dict = {"demo_task": {"test": [{"id": 0}]}}
        model.cache_hook = types.SimpleNamespace(add_partial=lambda *args, **kwargs: None)
        return model

    def test_generate_until_passes_video_metadata_and_kwargs_to_processor_with_fps(self):
        model = self._make_model(fps=2.5)
        metadata = _FakeVideoMetadata([0, 2, 4], fps=6.0, total_num_frames=5)
        video_tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        request = types.SimpleNamespace(
            args=("Describe the video", lambda doc: [{"role": "user", "content": []}], {}, 0, "demo_task", "test"),
        )

        with (
            patch("lmms_eval.models.chat.qwen2_5_vl.ChatMessages", _FakeChatMessages),
            patch("lmms_eval.models.chat.qwen2_5_vl.process_vision_info", return_value=(None, [(video_tensor, metadata)], {"do_sample_frames": False})),
            patch("lmms_eval.models.chat.qwen2_5_vl.decord", types.SimpleNamespace(VideoReader=_FakeVideoReader)),
            patch("lmms_eval.models.chat.qwen2_5_vl.log_metrics", lambda **kwargs: None),
        ):
            result = model.generate_until([request])

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, "final answer")
        self.assertEqual(_FakeChatMessages.last_video_kwargs["fps"], 2.5)

        processor_call = model.processor.calls[0]
        self.assertTrue(torch.equal(processor_call["videos"][0], video_tensor))
        self.assertIs(processor_call["video_metadata"][0], metadata)
        self.assertFalse(processor_call["do_sample_frames"])
        self.assertFalse(processor_call["do_resize"])
        self.assertAlmostEqual(processor_call["video_metadata"][0].sampled_fps, 3.6)

    def test_generate_until_keeps_sampled_metadata_in_sync_when_using_nframes(self):
        model = self._make_model(max_num_frames=3, fps=None)
        request = types.SimpleNamespace(
            args=("Describe the video", lambda doc: [{"role": "user", "content": []}], {}, 0, "demo_task", "test"),
        )

        def fake_process_vision_info(batched_messages, return_video_kwargs=False, image_patch_size=14, return_video_metadata=False):
            video_content = batched_messages[0][0]["content"][0]
            nframes = video_content["nframes"]
            sampled_indices = np.linspace(0, 4, nframes, dtype=int)
            video_tensor = torch.arange(nframes * 4, dtype=torch.float32).reshape(nframes, 4)
            metadata = _FakeVideoMetadata(sampled_indices.tolist(), fps=10.0, total_num_frames=5)
            return None, [(video_tensor, metadata)], {"do_sample_frames": False}

        with (
            patch("lmms_eval.models.chat.qwen2_5_vl.ChatMessages", _FakeChatMessages),
            patch("lmms_eval.models.chat.qwen2_5_vl.process_vision_info", side_effect=fake_process_vision_info),
            patch("lmms_eval.models.chat.qwen2_5_vl.decord", types.SimpleNamespace(VideoReader=_FakeVideoReader)),
            patch("lmms_eval.models.chat.qwen2_5_vl.log_metrics", lambda **kwargs: None),
        ):
            result = model.generate_until([request])

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, "final answer")
        self.assertEqual(_FakeChatMessages.last_video_kwargs["nframes"], 2)

        processor_call = model.processor.calls[0]
        metadata = processor_call["video_metadata"][0]
        self.assertEqual(processor_call["videos"][0].shape[0], 2)
        self.assertTrue(np.array_equal(metadata.frames_indices, np.array([0, 4])))
        self.assertEqual(len(metadata.frames_indices), processor_call["videos"][0].shape[0])
        self.assertAlmostEqual(metadata.sampled_fps, 4.0)
        self.assertFalse(processor_call["do_sample_frames"])
        self.assertFalse(processor_call["do_resize"])


if __name__ == "__main__":
    unittest.main()
