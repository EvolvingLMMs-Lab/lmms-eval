"""Compatibility tests for model-registry startup and legacy projections."""

import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from lmms_eval import models
from lmms_eval.models.registry_v2 import ModelRegistryV2


def _resolution_tuple(registry, model_name, *, force_simple=False):
    resolved = registry.resolve(model_name, force_simple=force_simple)
    return resolved.model_id, resolved.model_type, resolved.class_path


def test_builtin_manifests_preserve_all_model_resolutions():
    registry = ModelRegistryV2()
    registry.register_manifests(models._build_builtin_manifests())

    rows = [
        {
            "name": name,
            "default": _resolution_tuple(registry, name),
            "force_simple": _resolution_tuple(registry, name, force_simple=True),
        }
        for name in registry.list_model_names()
    ]
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))

    assert len(registry.list_canonical_model_ids()) == 117
    assert len(registry.list_model_names()) == 127
    assert hashlib.sha256(payload.encode()).hexdigest() == "45bb84f87177d5489d702387fb917f8942507d2cc75bc686334aa864893157a5"


def _registry_with_projection_fixtures():
    registry = ModelRegistryV2()
    registry.register_manifests(
        (
            models.ModelManifest(
                "builtin",
                simple_class_path="lmms_eval.models.simple.builtin.BuiltinSimple",
                chat_class_path="lmms_eval.models.chat.builtin.BuiltinChat",
                aliases=("builtin_alias",),
            ),
            models.ModelManifest(
                "external",
                simple_class_path="external.models.external.ExternalSimple",
                chat_class_path="external.models.external.ExternalChat",
                aliases=("external_alias",),
            ),
        ),
    )
    return registry


def test_legacy_views_project_manifests_without_importing_classes():
    registry = _registry_with_projection_fixtures()

    simple, chat, aliases, preferred = models._build_legacy_views(registry)

    assert dict(simple) == {
        "builtin": "BuiltinSimple",
        "external": "external.models.external.ExternalSimple",
    }
    assert dict(chat) == {
        "builtin": "BuiltinChat",
        "external": "external.models.external.ExternalChat",
    }
    assert dict(aliases) == {
        "builtin": ("builtin_alias",),
        "external": ("external_alias",),
    }
    assert dict(preferred) == {
        "builtin": "BuiltinChat",
        "external": "ExternalChat",
    }


def test_legacy_views_are_read_only():
    views = models._build_legacy_views(_registry_with_projection_fixtures())

    for view in views:
        try:
            view["new"] = "New"
        except TypeError:
            pass
        else:
            raise AssertionError(f"{type(view).__name__} accepted assignment")


def test_legacy_views_reflect_later_registration():
    registry = _registry_with_projection_fixtures()
    simple, chat, aliases, preferred = models._build_legacy_views(registry)

    registry.register_manifest(
        models.ModelManifest(
            "later",
            simple_class_path="plugin.models.later.LaterSimple",
            aliases=("later_alias",),
        ),
    )

    assert simple["later"] == "plugin.models.later.LaterSimple"
    assert "later" not in chat
    assert aliases["later"] == ("later_alias",)
    assert preferred["later"] == "LaterSimple"


def test_legacy_alias_view_exposes_only_current_owner_after_remap():
    registry = ModelRegistryV2()
    registry.register_manifest(models.ModelManifest("original", simple_class_path="pkg.Original", aliases=("shared",)))
    _, _, aliases, _ = models._build_legacy_views(registry)

    registry.register_manifest(models.ModelManifest("replacement", simple_class_path="pkg.Replacement", aliases=("shared",)), overwrite=True)

    assert "original" not in aliases
    assert aliases["replacement"] == ("shared",)


def test_discovery_does_not_import_backend_modules():
    code = "import sys; import lmms_eval.models; assert 'lmms_eval.models.simple.qwen3_vl' not in sys.modules; assert 'lmms_eval.models.chat.vllm' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_legacy_environment_plugins_isolate_failures(monkeypatch):
    registry = ModelRegistryV2()
    builtin_simple_before = models._BUILTIN_SIMPLE_MODELS.copy()

    def import_module(name):
        if name == "broken.models":
            raise RuntimeError("boom")
        if name == "healthy.models":
            return SimpleNamespace(AVAILABLE_MODELS={"external_simple": "ExternalSimple"})
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(models.importlib, "import_module", import_module)

    failures = models._load_legacy_plugin_models(registry, "broken,healthy")

    assert [(failure.source, failure.error_type, failure.message) for failure in failures] == [("legacy:broken", "RuntimeError", "boom")]
    assert registry.resolve("external_simple").class_path == "healthy.models.external_simple.ExternalSimple"
    assert models._BUILTIN_SIMPLE_MODELS == builtin_simple_before


def test_legacy_environment_plugin_preserves_existing_name_overwrite(monkeypatch):
    registry = ModelRegistryV2()
    registry.register_manifests(models._build_builtin_manifests())
    plugin = SimpleNamespace(AVAILABLE_MODELS={"qwen3_vl": "ReplacementQwen"})
    monkeypatch.setattr(models.importlib, "import_module", lambda name: plugin)

    failures = models._load_legacy_plugin_models(registry, "healthy")

    assert failures == ()
    assert registry.resolve("qwen3_vl", force_simple=True).class_path == "healthy.models.qwen3_vl.ReplacementQwen"
    assert registry.resolve("qwen3_vl").class_path == "lmms_eval.models.chat.qwen3_vl.Qwen3_VL"


def test_legacy_environment_plugins_validate_each_package_atomically(monkeypatch):
    registry = ModelRegistryV2()
    modules = {
        "invalid_pair.models": SimpleNamespace(AVAILABLE_MODELS={"valid": "Valid", "": "Empty"}),
        "not_mapping.models": SimpleNamespace(AVAILABLE_MODELS=[("other", "Other")]),
        "healthy.models": SimpleNamespace(AVAILABLE_MODELS={"healthy": "Healthy"}),
    }
    monkeypatch.setattr(models.importlib, "import_module", modules.__getitem__)

    failures = models._load_legacy_plugin_models(registry, "invalid_pair,not_mapping,healthy")

    assert [(failure.source, failure.error_type) for failure in failures] == [
        ("legacy:invalid_pair", "TypeError"),
        ("legacy:not_mapping", "TypeError"),
    ]
    with pytest.raises(ValueError, match="valid"):
        registry.resolve("valid")
    assert registry.resolve("healthy").class_path == "healthy.models.healthy.Healthy"
