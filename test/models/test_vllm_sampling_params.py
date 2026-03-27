import types
import unittest
from unittest.mock import patch

from lmms_eval.models.chat.vllm import VLLM as ChatVLLM
from lmms_eval.models.chat.vllm_generate import VLLMGenerate
from lmms_eval.models.simple.vllm import VLLM as SimpleVLLM


class _FakeChatMessages:
    def __init__(self, messages):
        self.messages = messages

    def to_openai_messages(self, video_kwargs=None):
        return [{"role": "user", "content": [{"type": "text", "text": "Describe the input"}]}]

    def to_hf_messages(self, video_kwargs=None):
        return [{"role": "user", "content": [{"type": "text", "text": "Describe the input"}]}]

    def extract_media(self):
        return [], [], []


class _FakeProcessor:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "prompt"


class TestVllmSamplingParams(unittest.TestCase):
    def _make_simple_model(self):
        model = SimpleVLLM.__new__(SimpleVLLM)
        model.max_new_tokens = 1024
        return model

    def _make_chat_model(self, cls):
        model = cls.__new__(cls)
        model.max_new_tokens = 1024
        model.max_pixels = 1024
        model.min_image_pixels = 28
        model.max_frame_num = 32
        model.fps = None
        model.nframes = 8
        model.processor = _FakeProcessor()
        model.task_dict = {"demo_task": {"test": [{"id": 0}]}}
        return model

    def test_build_sampling_params_clamps_zero_top_p(self):
        model = self._make_simple_model()

        params = model._build_sampling_params_dict({"max_new_tokens": 128, "temperature": 0, "top_p": 0})

        self.assertEqual(params["top_p"], 1.0)

    def test_build_sampling_params_preserves_valid_top_p(self):
        model = self._make_simple_model()

        params = model._build_sampling_params_dict({"max_new_tokens": 128, "temperature": 0, "top_p": 0.95})

        self.assertEqual(params["top_p"], 0.95)

    def test_chat_make_one_request_clamps_zero_top_p(self):
        model = self._make_chat_model(ChatVLLM)
        request = types.SimpleNamespace(
            arguments=("Describe the input", lambda doc: [{"role": "user", "content": []}], {"temperature": 0, "top_p": 0}, 0, "demo_task", "test"),
        )

        with patch("lmms_eval.models.chat.vllm.ChatMessages", _FakeChatMessages):
            _, params = model.make_one_request(request)

        self.assertEqual(params["top_p"], 1.0)

    def test_generate_make_one_request_clamps_zero_top_p(self):
        model = self._make_chat_model(VLLMGenerate)
        request = types.SimpleNamespace(
            arguments=("Describe the input", lambda doc: [{"role": "user", "content": []}], {"temperature": 0, "top_p": 0}, 0, "demo_task", "test"),
        )

        with patch("lmms_eval.models.chat.vllm_generate.ChatMessages", _FakeChatMessages):
            _, params = model.make_one_request(request)

        self.assertEqual(params["top_p"], 1.0)


if __name__ == "__main__":
    unittest.main()
