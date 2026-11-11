import inspect
import unittest
from unittest import mock

from lmms_eval.models import get_model, get_model_manifest
from lmms_eval.models.chat.minicpm_ocr import (
    MiniCPMOcr,
    _require_transformers_version,
)


class TestMiniCPMOcrRegistration(unittest.TestCase):
    def test_registry_resolves_chat_adapter(self):
        manifest = get_model_manifest("minicpm_ocr")

        self.assertEqual(manifest.chat_class_path, "lmms_eval.models.chat.minicpm_ocr.MiniCPMOcr")
        self.assertIsNone(manifest.simple_class_path)
        self.assertIs(get_model("minicpm_ocr"), MiniCPMOcr)
        self.assertFalse(MiniCPMOcr.is_simple)

    def test_defaults_target_minicpm_ocr_checkpoint(self):
        params = inspect.signature(MiniCPMOcr.__init__).parameters

        self.assertEqual(params["pretrained"].default, "openbmb/MiniCPM-OCR-1.0")
        self.assertEqual(params["downsample_mode"].default, "4x")
        self.assertFalse(params["enable_thinking"].default)


class TestMiniCPMOcrGenerationHelpers(unittest.TestCase):
    def make_model(self):
        model = object.__new__(MiniCPMOcr)
        model.use_cache = True
        model._tokenizer = type("Tokenizer", (), {"pad_token_id": 0})()
        return model

    def test_requires_transformers_with_native_minicpm_ocr(self):
        with mock.patch("transformers.__version__", "5.22.2"):
            _require_transformers_version()

        with mock.patch("transformers.__version__", "5.21.1"):
            with self.assertRaisesRegex(ImportError, "5.22.0"):
                _require_transformers_version()

    def test_normalize_until_accepts_string_and_drops_empty_entries(self):
        self.assertEqual(MiniCPMOcr._normalize_until(None), [])
        self.assertEqual(MiniCPMOcr._normalize_until("END"), ["END"])
        self.assertEqual(MiniCPMOcr._normalize_until(["END", "", None]), ["END"])
        with self.assertRaises(ValueError):
            MiniCPMOcr._normalize_until(1)

    def test_greedy_generation_ignores_sampling_settings(self):
        model = self.make_model()

        kwargs = model._build_generate_kwargs({"max_new_tokens": 32, "temperature": 0, "top_p": 0.9})

        self.assertEqual(kwargs["max_new_tokens"], 32)
        self.assertFalse(kwargs["do_sample"])
        self.assertNotIn("temperature", kwargs)
        self.assertNotIn("top_p", kwargs)

    def test_sampling_generation_keeps_sampling_settings(self):
        model = self.make_model()

        kwargs = model._build_generate_kwargs({"max_new_tokens": 16, "temperature": 0.7, "top_p": 0.8, "top_k": 20})

        self.assertTrue(kwargs["do_sample"])
        self.assertEqual(kwargs["temperature"], 0.7)
        self.assertEqual(kwargs["top_p"], 0.8)
        self.assertEqual(kwargs["top_k"], 20)

    def test_until_stops_at_earliest_marker(self):
        model = self.make_model()

        answer = model._truncate_until("first\n\nsecond<|im_end|>", model._normalize_until(["\n\n", "<|im_end|>"]))

        self.assertEqual(answer, "first")

    def test_missing_until_returns_answer_unchanged(self):
        model = self.make_model()

        self.assertEqual(model._truncate_until("plain answer", []), "plain answer")


if __name__ == "__main__":
    unittest.main()
