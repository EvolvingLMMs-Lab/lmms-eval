"""Unit tests for model registry v2 resolution semantics."""

import importlib.util
import pathlib
import sys
import unittest

_REGISTRY_PATH = pathlib.Path(__file__).resolve().parents[2] / "lmms_eval" / "models" / "registry_v2.py"
_SPEC = importlib.util.spec_from_file_location("registry_v2_for_tests", _REGISTRY_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

ModelManifest = _MODULE.ModelManifest
ModelRegistryV2 = _MODULE.ModelRegistryV2


class TestModelRegistryV2(unittest.TestCase):
    def test_chat_precedence_and_force_simple(self):
        registry = ModelRegistryV2()
        registry.register_manifest(
            ModelManifest(
                model_id="demo",
                simple_class_path="pkg.simple.DemoSimple",
                chat_class_path="pkg.chat.DemoChat",
            ),
        )

        resolved_default = registry.resolve("demo")
        self.assertEqual(resolved_default.model_type, "chat")
        self.assertEqual(resolved_default.class_path, "pkg.chat.DemoChat")

        resolved_force_simple = registry.resolve("demo", force_simple=True)
        self.assertEqual(resolved_force_simple.model_type, "simple")
        self.assertEqual(resolved_force_simple.class_path, "pkg.simple.DemoSimple")

    def test_force_simple_ignored_when_simple_missing(self):
        registry = ModelRegistryV2()
        registry.register_manifest(
            ModelManifest(
                model_id="chat_only",
                chat_class_path="pkg.chat.ChatOnly",
            ),
        )

        resolved = registry.resolve("chat_only", force_simple=True)
        self.assertEqual(resolved.model_type, "chat")
        self.assertEqual(resolved.class_path, "pkg.chat.ChatOnly")

    def test_alias_resolution(self):
        registry = ModelRegistryV2()
        registry.register_manifest(
            ModelManifest(
                model_id="api",
                chat_class_path="pkg.chat.Api",
                aliases=("api_chat",),
            ),
        )

        resolved = registry.resolve("api_chat")
        self.assertEqual(resolved.model_id, "api")
        self.assertEqual(resolved.requested_name, "api_chat")


class TestRepresentativeManifestSemantics(unittest.TestCase):
    def test_representative_aliases_are_resolvable(self):
        registry = ModelRegistryV2()
        registry.register_manifest(
            ModelManifest(
                model_id="vllm",
                simple_class_path="lmms_eval.models.simple.vllm.VLLM",
                chat_class_path="lmms_eval.models.chat.vllm.VLLM",
                aliases=("vllm_chat",),
            ),
        )
        registry.register_manifest(
            ModelManifest(
                model_id="openai_compatible",
                simple_class_path="lmms_eval.models.simple.openai_compatible.OpenAICompatible",
                chat_class_path="lmms_eval.models.chat.openai_compatible.OpenAICompatible",
                aliases=("openai_compatible_chat",),
            ),
        )
        registry.register_manifest(
            ModelManifest(
                model_id="sglang",
                chat_class_path="lmms_eval.models.chat.sglang.Sglang",
                aliases=("sglang_runtime",),
            ),
        )

        self.assertEqual(registry.resolve("vllm_chat").model_id, "vllm")
        self.assertEqual(
            registry.resolve("openai_compatible_chat").model_id,
            "openai_compatible",
        )
        self.assertEqual(registry.resolve("sglang_runtime").model_id, "sglang")

    def test_representative_force_simple_behavior(self):
        registry = ModelRegistryV2()
        registry.register_manifest(
            ModelManifest(
                model_id="vllm",
                simple_class_path="lmms_eval.models.simple.vllm.VLLM",
                chat_class_path="lmms_eval.models.chat.vllm.VLLM",
            ),
        )
        registry.register_manifest(
            ModelManifest(
                model_id="sglang",
                chat_class_path="lmms_eval.models.chat.sglang.Sglang",
            ),
        )

        self.assertEqual(registry.resolve("vllm").model_type, "chat")
        self.assertEqual(registry.resolve("vllm", force_simple=True).model_type, "simple")
        self.assertEqual(registry.resolve("sglang", force_simple=True).model_type, "chat")


if __name__ == "__main__":
    unittest.main()
