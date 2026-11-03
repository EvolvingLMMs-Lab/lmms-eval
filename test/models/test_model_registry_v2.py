"""Unit tests for model registry v2 resolution semantics."""

import importlib.util
import pathlib
import sys
import unittest
from unittest.mock import patch

_REGISTRY_PATH = pathlib.Path(__file__).resolve().parents[2] / "lmms_eval" / "models" / "registry_v2.py"
_SPEC = importlib.util.spec_from_file_location("registry_v2_for_tests", _REGISTRY_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

ModelManifest = _MODULE.ModelManifest
ModelRegistryV2 = _MODULE.ModelRegistryV2


class _FakeEntryPoint:
    def __init__(self, name, value, payload=None, error=None):
        self.name = name
        self.value = value
        self.payload = payload
        self.error = error

    def load(self):
        if self.error:
            raise self.error
        return self.payload


class _BrokenIterable:
    def __iter__(self):
        raise RuntimeError("iteration failed")


class _BrokenSortEntryPoint:
    @property
    def name(self):
        raise RuntimeError("sort key failed")


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

    def test_alias_conflict_does_not_partially_register_model(self):
        registry = ModelRegistryV2()
        registry.register_manifest(
            ModelManifest("owner", simple_class_path="pkg.Owner", aliases=("shared",)),
        )

        with self.assertRaisesRegex(ValueError, "shared"):
            registry.register_manifest(
                ModelManifest("contender", simple_class_path="pkg.Contender", aliases=("shared",)),
            )

        self.assertEqual(registry.list_canonical_model_ids(), ["owner"])
        self.assertEqual(registry.resolve("shared").model_id, "owner")

    def test_same_id_fills_missing_lane_and_deduplicates_alias(self):
        registry = ModelRegistryV2()
        registry.register_manifest(
            ModelManifest("dual", simple_class_path="pkg.Simple", aliases=("dual_api",)),
        )
        registry.register_manifest(
            ModelManifest("dual", chat_class_path="pkg.Chat", aliases=("dual_api",)),
        )

        self.assertEqual(
            registry.get_manifest("dual"),
            ModelManifest("dual", "pkg.Simple", "pkg.Chat", ("dual_api",)),
        )

    def test_conflicting_path_preserves_original(self):
        registry = ModelRegistryV2()
        registry.register_manifest(ModelManifest("demo", simple_class_path="pkg.Original"))

        with self.assertRaisesRegex(ValueError, "Conflicting simple_class_path"):
            registry.register_manifest(ModelManifest("demo", simple_class_path="pkg.Replacement"))

        self.assertEqual(registry.resolve("demo").class_path, "pkg.Original")

    def test_list_manifests_returns_canonical_entries_without_importing_paths(self):
        registry = ModelRegistryV2()
        registry.register_manifest(ModelManifest("zeta", chat_class_path="missing.module.Zeta"))
        registry.register_manifest(ModelManifest("alpha", simple_class_path="missing.module.Alpha"))

        self.assertEqual(
            registry.list_manifests(),
            [
                ModelManifest("alpha", simple_class_path="missing.module.Alpha"),
                ModelManifest("zeta", chat_class_path="missing.module.Zeta"),
            ],
        )

    def test_overwrite_transfers_alias_and_removes_stale_owner(self):
        registry = ModelRegistryV2()
        registry.register_manifest(ModelManifest("original", simple_class_path="pkg.Original", aliases=("shared",)))

        registry.register_manifest(ModelManifest("replacement", simple_class_path="pkg.Replacement", aliases=("shared",)), overwrite=True)

        self.assertEqual(registry.resolve("original").model_id, "original")
        self.assertEqual(registry.resolve("replacement").model_id, "replacement")
        self.assertEqual(registry.resolve("shared").model_id, "replacement")
        self.assertEqual(registry.get_manifest("original").aliases, ())

    def test_overwrite_canonical_id_takes_ownership_from_old_alias(self):
        registry = ModelRegistryV2()
        registry.register_manifest(ModelManifest("original", simple_class_path="pkg.Original", aliases=("replacement",)))

        registry.register_manifest(ModelManifest("replacement", simple_class_path="pkg.Replacement"), overwrite=True)

        self.assertEqual(registry.resolve("original").model_id, "original")
        self.assertEqual(registry.resolve("replacement").model_id, "replacement")
        self.assertEqual(registry.get_manifest("original").aliases, ())

    def test_overwrite_alias_cannot_shadow_another_canonical_id(self):
        registry = ModelRegistryV2()
        registry.register_manifest(ModelManifest("protected", simple_class_path="pkg.Protected"))

        with self.assertRaisesRegex(ValueError, "canonical model id"):
            registry.register_manifests(
                (
                    ModelManifest("innocent", simple_class_path="pkg.Innocent"),
                    ModelManifest("contender", simple_class_path="pkg.Contender", aliases=("protected",)),
                ),
                overwrite=True,
            )

        self.assertEqual(registry.resolve("protected").model_id, "protected")
        with self.assertRaisesRegex(ValueError, "innocent"):
            registry.resolve("innocent")
        with self.assertRaisesRegex(ValueError, "contender"):
            registry.resolve("contender")


class TestEntryPointRegistration(unittest.TestCase):
    def _assert_discovery_failure(self, registry, failures, message):
        self.assertEqual(
            [(failure.source, failure.error_type, failure.message) for failure in failures],
            [("entrypoints:lmms_eval.models", "RuntimeError", message)],
        )
        self.assertEqual(registry.resolve("builtin").class_path, "pkg.Builtin")

    def test_entry_point_enumeration_failure_is_reported(self):
        registry = ModelRegistryV2()
        registry.register_manifest(ModelManifest("builtin", simple_class_path="pkg.Builtin"))

        with patch.object(_MODULE, "entry_points", side_effect=RuntimeError("enumeration failed")):
            failures = registry.load_entrypoint_manifests()

        self._assert_discovery_failure(registry, failures, "enumeration failed")

    def test_entry_point_selection_iteration_failure_is_reported(self):
        registry = ModelRegistryV2()
        registry.register_manifest(ModelManifest("builtin", simple_class_path="pkg.Builtin"))
        registry._select_entry_points = lambda group: _BrokenIterable()

        failures = registry.load_entrypoint_manifests()

        self._assert_discovery_failure(registry, failures, "iteration failed")

    def test_entry_point_sort_key_failure_is_reported(self):
        registry = ModelRegistryV2()
        registry.register_manifest(ModelManifest("builtin", simple_class_path="pkg.Builtin"))
        registry._select_entry_points = lambda group: [_BrokenSortEntryPoint()]

        failures = registry.load_entrypoint_manifests()

        self._assert_discovery_failure(registry, failures, "sort key failed")

    def test_entry_point_failure_is_reported_and_later_plugin_loads(self):
        registry = ModelRegistryV2()
        registry._select_entry_points = lambda group: [
            _FakeEntryPoint("broken", "broken:models", error=RuntimeError("boom")),
            _FakeEntryPoint(
                "healthy",
                "healthy:models",
                ModelManifest("external", chat_class_path="healthy.Chat"),
            ),
        ]

        failures = registry.load_entrypoint_manifests()

        self.assertEqual(
            [(failure.source, failure.error_type, failure.message) for failure in failures],
            [("broken (broken:models)", "RuntimeError", "boom")],
        )
        self.assertEqual(registry.resolve("external").class_path, "healthy.Chat")

    def test_entry_point_batch_rollback_preserves_existing_and_loads_later_plugin(self):
        registry = ModelRegistryV2()
        registry.register_manifest(ModelManifest("existing", simple_class_path="pkg.Original"))
        registry._select_entry_points = lambda group: [
            _FakeEntryPoint(
                "conflicting",
                "conflicting:models",
                (
                    ModelManifest("rolled_back", simple_class_path="pkg.RolledBack"),
                    ModelManifest("existing", simple_class_path="pkg.Conflict"),
                ),
            ),
            _FakeEntryPoint(
                "healthy",
                "healthy:models",
                ModelManifest("healthy", simple_class_path="pkg.Healthy"),
            ),
        ]

        failures = registry.load_entrypoint_manifests()

        self.assertEqual(
            [(failure.source, failure.error_type) for failure in failures],
            [("conflicting (conflicting:models)", "ValueError")],
        )
        self.assertEqual(registry.resolve("existing").class_path, "pkg.Original")
        with self.assertRaisesRegex(ValueError, "rolled_back"):
            registry.resolve("rolled_back")
        self.assertEqual(registry.resolve("healthy").class_path, "pkg.Healthy")


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
                model_id="openai",
                simple_class_path="lmms_eval.models.simple.openai.OpenAICompatible",
                chat_class_path="lmms_eval.models.chat.openai.OpenAICompatible",
                aliases=("openai_compatible", "openai_compatible_chat"),
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
            "openai",
        )
        self.assertEqual(
            registry.resolve("openai_compatible").model_id,
            "openai",
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


ResolvedModel = _MODULE.ResolvedModel


class TestValidateModelClass(unittest.TestCase):
    def setUp(self):
        import types

        self._fake_model_module = types.ModuleType("lmms_eval.api.model")

        class FakeLmms:
            is_simple = True

        self._fake_model_module.lmms = FakeLmms
        self._original = sys.modules.get("lmms_eval.api.model")
        sys.modules["lmms_eval.api.model"] = self._fake_model_module
        self.FakeLmms = FakeLmms

    def tearDown(self):
        if self._original is not None:
            sys.modules["lmms_eval.api.model"] = self._original
        else:
            sys.modules.pop("lmms_eval.api.model", None)

    def _resolved(self, model_type, class_path="pkg.Foo"):
        return ResolvedModel(
            requested_name="test",
            model_id="test",
            model_type=model_type,
            class_path=class_path,
        )

    def test_valid_simple_class(self):
        class SimpleModel(self.FakeLmms):
            is_simple = True

        ModelRegistryV2._validate_model_class(SimpleModel, self._resolved("simple"))

    def test_valid_chat_class(self):
        class ChatModel(self.FakeLmms):
            is_simple = False

        ModelRegistryV2._validate_model_class(ChatModel, self._resolved("chat"))

    def test_not_a_subclass_raises(self):
        class NotAModel:
            pass

        with self.assertRaises(TypeError) as ctx:
            ModelRegistryV2._validate_model_class(NotAModel, self._resolved("simple", "pkg.NotAModel"))
        self.assertIn("not a subclass", str(ctx.exception))

    def test_not_a_class_raises(self):
        with self.assertRaises(TypeError) as ctx:
            ModelRegistryV2._validate_model_class("not_a_class", self._resolved("simple", "pkg.oops"))
        self.assertIn("not a subclass", str(ctx.exception))

    def test_chat_resolved_but_is_simple_true_raises(self):
        class BadChat(self.FakeLmms):
            is_simple = True

        with self.assertRaises(TypeError) as ctx:
            ModelRegistryV2._validate_model_class(BadChat, self._resolved("chat"))
        self.assertIn("resolved as chat", str(ctx.exception))
        self.assertIn("is_simple is True", str(ctx.exception))

    def test_simple_resolved_but_is_simple_false_raises(self):
        class BadSimple(self.FakeLmms):
            is_simple = False

        with self.assertRaises(TypeError) as ctx:
            ModelRegistryV2._validate_model_class(BadSimple, self._resolved("simple"))
        self.assertIn("resolved as simple", str(ctx.exception))
        self.assertIn("is_simple is False", str(ctx.exception))

    def test_missing_is_simple_defaults_true(self):
        class NoFlag(self.FakeLmms):
            pass

        ModelRegistryV2._validate_model_class(NoFlag, self._resolved("simple"))
        with self.assertRaises(TypeError):
            ModelRegistryV2._validate_model_class(NoFlag, self._resolved("chat"))


if __name__ == "__main__":
    unittest.main()
