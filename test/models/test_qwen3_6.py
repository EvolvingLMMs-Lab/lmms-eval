import inspect
import unittest

from lmms_eval.models import get_model, get_model_manifest
from lmms_eval.models.chat.qwen3_6 import Qwen3_6 as Qwen3_6_Chat
from lmms_eval.models.simple.qwen3_5 import Qwen3_5
from lmms_eval.models.simple.qwen3_6 import Qwen3_6 as Qwen3_6_Simple


class TestQwen36Registration(unittest.TestCase):
    def test_defaults_target_official_moe_checkpoint(self):
        pretrained = inspect.signature(Qwen3_6_Simple.__init__).parameters["pretrained"]

        self.assertEqual(pretrained.default, "Qwen/Qwen3.6-35B-A3B")
        self.assertEqual(
            Qwen3_6_Simple.DEFAULT_GEN_KWARGS,
            {
                "max_new_tokens": 1024,
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 20,
            },
        )
        self.assertTrue(issubclass(Qwen3_6_Simple, Qwen3_5))

    def test_registry_prefers_chat_and_retains_simple_adapter(self):
        manifest = get_model_manifest("qwen3_6")

        self.assertEqual(manifest.chat_class_path, "lmms_eval.models.chat.qwen3_6.Qwen3_6")
        self.assertEqual(manifest.simple_class_path, "lmms_eval.models.simple.qwen3_6.Qwen3_6")
        self.assertIs(get_model("qwen3_6"), Qwen3_6_Chat)
        self.assertIs(get_model("qwen3_6", force_simple=True), Qwen3_6_Simple)
        self.assertFalse(Qwen3_6_Chat.is_simple)


if __name__ == "__main__":
    unittest.main()
