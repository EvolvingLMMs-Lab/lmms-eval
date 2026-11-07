import inspect
import unittest
from types import SimpleNamespace
from unittest import mock

import transformers

from lmms_eval.models import get_model, get_model_manifest
from lmms_eval.models.chat.qwen3_8 import Qwen3_8 as Qwen3_8_Chat
from lmms_eval.models.simple import qwen3_vl
from lmms_eval.models.simple.qwen3_5 import Qwen3_5
from lmms_eval.models.simple.qwen3_8 import Qwen3_8 as Qwen3_8_Simple


class TestQwen38Registration(unittest.TestCase):
    def test_defaults_target_flash_next_checkpoint(self):
        pretrained = inspect.signature(Qwen3_8_Simple.__init__).parameters["pretrained"]

        self.assertEqual(pretrained.default, "Qwen/Qwen3.8-Flash-Next")
        self.assertEqual(
            Qwen3_8_Simple.DEFAULT_GEN_KWARGS,
            {
                "max_new_tokens": 1024,
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 20,
            },
        )
        self.assertTrue(issubclass(Qwen3_8_Simple, Qwen3_5))

    def test_registry_prefers_chat_and_retains_simple_adapter(self):
        manifest = get_model_manifest("qwen3_8")

        self.assertEqual(manifest.chat_class_path, "lmms_eval.models.chat.qwen3_8.Qwen3_8")
        self.assertEqual(manifest.simple_class_path, "lmms_eval.models.simple.qwen3_8.Qwen3_8")
        self.assertIs(get_model("qwen3_8"), Qwen3_8_Chat)
        self.assertIs(get_model("qwen3_8", force_simple=True), Qwen3_8_Simple)
        self.assertFalse(Qwen3_8_Chat.is_simple)


@unittest.skipUnless(hasattr(transformers, "Qwen4ExpForConditionalGeneration"), "transformers without qwen4_exp")
class TestQwen38ModelClass(unittest.TestCase):
    def test_qwen4_exp_config_resolves_to_qwen4_exp_class(self):
        with mock.patch.object(qwen3_vl.AutoConfig, "from_pretrained", return_value=SimpleNamespace(model_type="qwen4_exp")):
            model_cls, dtype_key = qwen3_vl._resolve_model_class("Qwen/Qwen3.8-Flash-Next", is_moe=False)

        self.assertIs(model_cls, transformers.Qwen4ExpForConditionalGeneration)
        self.assertEqual(dtype_key, "dtype")


if __name__ == "__main__":
    unittest.main()
