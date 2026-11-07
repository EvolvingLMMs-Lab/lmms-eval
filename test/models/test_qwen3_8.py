import inspect
import unittest
from types import SimpleNamespace
from unittest import mock

import transformers

from lmms_eval.models import get_model, get_model_manifest
from lmms_eval.models.chat.qwen3_8 import Qwen3_8 as Qwen3_8_Chat
from lmms_eval.models.chat.qwen3_8_flash_next import (
    Qwen3_8_FlashNext as Qwen3_8_FlashNext_Chat,
)
from lmms_eval.models.simple import qwen3_vl
from lmms_eval.models.simple.qwen3_5 import Qwen3_5
from lmms_eval.models.simple.qwen3_8 import Qwen3_8 as Qwen3_8_Simple
from lmms_eval.models.simple.qwen3_8_flash_next import (
    Qwen3_8_FlashNext as Qwen3_8_FlashNext_Simple,
)

THINKING_GEN_KWARGS = {
    "max_new_tokens": 1024,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
}


def default_pretrained(model_cls):
    return inspect.signature(model_cls.__init__).parameters["pretrained"].default


class TestQwen38Registration(unittest.TestCase):
    def test_dense_defaults_target_27b_checkpoint(self):
        self.assertEqual(default_pretrained(Qwen3_8_Simple), "Qwen/Qwen3.8-27B")
        self.assertEqual(Qwen3_8_Simple.DEFAULT_GEN_KWARGS, THINKING_GEN_KWARGS)
        self.assertTrue(issubclass(Qwen3_8_Simple, Qwen3_5))

    def test_flash_next_defaults_target_flash_next_checkpoint(self):
        self.assertEqual(default_pretrained(Qwen3_8_FlashNext_Simple), "Qwen/Qwen3.8-Flash-Next")
        self.assertEqual(Qwen3_8_FlashNext_Simple.DEFAULT_GEN_KWARGS, THINKING_GEN_KWARGS)
        self.assertTrue(issubclass(Qwen3_8_FlashNext_Simple, Qwen3_5))
        self.assertFalse(issubclass(Qwen3_8_FlashNext_Simple, Qwen3_8_Simple))

    def test_registry_prefers_chat_and_retains_simple_adapter(self):
        cases = [
            ("qwen3_8", "qwen3_8.Qwen3_8", Qwen3_8_Chat, Qwen3_8_Simple),
            ("qwen3_8_flash_next", "qwen3_8_flash_next.Qwen3_8_FlashNext", Qwen3_8_FlashNext_Chat, Qwen3_8_FlashNext_Simple),
        ]
        for name, class_path, chat_cls, simple_cls in cases:
            with self.subTest(model=name):
                manifest = get_model_manifest(name)

                self.assertEqual(manifest.chat_class_path, f"lmms_eval.models.chat.{class_path}")
                self.assertEqual(manifest.simple_class_path, f"lmms_eval.models.simple.{class_path}")
                self.assertIs(get_model(name), chat_cls)
                self.assertIs(get_model(name, force_simple=True), simple_cls)
                self.assertFalse(chat_cls.is_simple)


@unittest.skipUnless(hasattr(transformers, "Qwen4ExpForConditionalGeneration"), "transformers without qwen4_exp")
class TestQwen38ModelClass(unittest.TestCase):
    def resolve(self, model_type):
        with mock.patch.object(qwen3_vl.AutoConfig, "from_pretrained", return_value=SimpleNamespace(model_type=model_type)):
            return qwen3_vl._resolve_model_class("Qwen/Qwen3.8", is_moe=False)

    def test_flash_next_config_resolves_to_qwen4_exp_class(self):
        model_cls, dtype_key = self.resolve("qwen4_exp")

        self.assertIs(model_cls, transformers.Qwen4ExpForConditionalGeneration)
        self.assertEqual(dtype_key, "dtype")

    def test_dense_config_resolves_to_qwen3_5_class(self):
        model_cls, dtype_key = self.resolve("qwen3_5")

        self.assertIs(model_cls, transformers.Qwen3_5ForConditionalGeneration)
        self.assertEqual(dtype_key, "torch_dtype")


if __name__ == "__main__":
    unittest.main()
