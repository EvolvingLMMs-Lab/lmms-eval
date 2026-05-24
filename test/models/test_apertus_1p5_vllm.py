import unittest

from lmms_eval.models.simple.vllm import VLLM


class _NewVLLMClient:
    def chat(self, messages, sampling_params=None, tokenization_kwargs=None):
        return []


class _OldVLLMClient:
    def chat(self, messages, sampling_params=None):
        return []


class TestApertus1p5VLLM(unittest.TestCase):
    def test_generic_vllm_does_not_override_chat_tokenization(self):
        model = VLLM.__new__(VLLM)
        model.client = _NewVLLMClient()

        self.assertEqual(model._chat_tokenization_kwargs(), {})

    def test_apertus_vllm_disables_special_tokens_when_supported(self):
        from lmms_eval.models.simple.apertus_1p5_vllm import Apertus1p5VLLM

        model = Apertus1p5VLLM.__new__(Apertus1p5VLLM)
        model.client = _NewVLLMClient()

        self.assertEqual(
            model._chat_tokenization_kwargs(),
            {"tokenization_kwargs": {"add_special_tokens": False}},
        )

    def test_apertus_vllm_leaves_old_vllm_signatures_unchanged(self):
        from lmms_eval.models.simple.apertus_1p5_vllm import Apertus1p5VLLM

        model = Apertus1p5VLLM.__new__(Apertus1p5VLLM)
        model.client = _OldVLLMClient()

        self.assertEqual(model._chat_tokenization_kwargs(), {})

    def test_registry_exposes_apertus_vllm_wrapper(self):
        from lmms_eval.models import AVAILABLE_CHAT_TEMPLATE_MODELS, AVAILABLE_SIMPLE_MODELS, get_model

        self.assertEqual(AVAILABLE_SIMPLE_MODELS["apertus_1p5_vllm"], "Apertus1p5VLLM")
        self.assertEqual(AVAILABLE_CHAT_TEMPLATE_MODELS["apertus_1p5_vllm"], "Apertus1p5VLLM")
        self.assertEqual(get_model("apertus_1p5_vllm").__module__, "lmms_eval.models.chat.apertus_1p5_vllm")
        self.assertEqual(get_model("apertus_1p5_vllm", force_simple=True).__module__, "lmms_eval.models.simple.apertus_1p5_vllm")

    def test_chat_apertus_vllm_disables_special_tokens_when_supported(self):
        from lmms_eval.models.chat.apertus_1p5_vllm import Apertus1p5VLLM

        model = Apertus1p5VLLM.__new__(Apertus1p5VLLM)
        model.client = _NewVLLMClient()

        self.assertEqual(
            model._chat_tokenization_kwargs(),
            {"tokenization_kwargs": {"add_special_tokens": False}},
        )


if __name__ == "__main__":
    unittest.main()
