from __future__ import annotations

import importlib
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Iterable, Literal

ModelType = Literal["simple", "chat"]


@dataclass(frozen=True)
class ModelManifest:
    """Declarative model registration payload for model loading.

    A single manifest can expose a chat implementation, a simple implementation,
    or both for the same `model_id`.
    """

    model_id: str
    simple_class_path: str | None = None
    chat_class_path: str | None = None
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.simple_class_path is None and self.chat_class_path is None:
            raise ValueError(
                f"ModelManifest('{self.model_id}') requires at least one class path",
            )

        normalized_aliases = tuple(alias for alias in dict.fromkeys(self.aliases) if alias and alias != self.model_id)
        object.__setattr__(self, "aliases", normalized_aliases)


@dataclass(frozen=True)
class ResolvedModel:
    """Resolution result for one requested model name."""

    requested_name: str
    model_id: str
    model_type: ModelType
    class_path: str

    @property
    def class_name(self) -> str:
        return self.class_path.rsplit(".", 1)[-1]


class ModelRegistryV2:
    """Canonical model registry with aliasing and typed resolution semantics."""

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self._manifests: dict[str, ModelManifest] = {}
        self._alias_to_model_id: dict[str, str] = {}

    def register_manifest(
        self,
        manifest: ModelManifest,
        *,
        overwrite: bool = False,
    ) -> None:
        """Register a manifest and all aliases.

        When `overwrite=False`, conflicting class paths for an existing model id
        raise `ValueError`.
        """

        merged = self._merge_manifest(self._manifests.get(manifest.model_id), manifest, overwrite=overwrite)
        self._manifests[manifest.model_id] = merged

        names = (merged.model_id, *merged.aliases)
        for name in names:
            existing = self._alias_to_model_id.get(name)
            if existing and existing != merged.model_id and not overwrite:
                raise ValueError(
                    f"Alias '{name}' already points to '{existing}', cannot remap to '{merged.model_id}'",
                )
            self._alias_to_model_id[name] = merged.model_id

    def resolve(self, model_name: str, force_simple: bool = False) -> ResolvedModel:
        """Resolve a model name to one concrete implementation class path."""

        model_id = self._require_model_id(model_name)

        manifest = self._manifests[model_id]
        if force_simple and manifest.simple_class_path is not None:
            model_type = "simple"
            class_path = manifest.simple_class_path
        elif manifest.chat_class_path is not None:
            model_type = "chat"
            class_path = manifest.chat_class_path
        else:
            assert manifest.simple_class_path is not None
            model_type = "simple"
            class_path = manifest.simple_class_path

        return ResolvedModel(
            requested_name=model_name,
            model_id=model_id,
            model_type=model_type,
            class_path=class_path,
        )

    def get_model_class(self, model_name: str, force_simple: bool = False):
        """Resolve, import, and validate a model class.

        Validates that the loaded class is a subclass of ``lmms`` and that its
        ``is_simple`` flag is consistent with the resolved model type.
        """

        resolved = self.resolve(model_name, force_simple=force_simple)
        module_name, cls_name = resolved.class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, cls_name)
        self._validate_model_class(cls, resolved)
        return cls

    @staticmethod
    def _validate_model_class(cls: type, resolved: ResolvedModel) -> None:
        from lmms_eval.api.model import lmms

        if not (isinstance(cls, type) and issubclass(cls, lmms)):
            raise TypeError(
                f"Model class '{resolved.class_path}' is not a subclass of lmms",
            )

        cls_is_simple = getattr(cls, "is_simple", True)
        if resolved.model_type == "chat" and cls_is_simple:
            raise TypeError(
                f"Model '{resolved.model_id}' resolved as chat but " f"{cls.__name__}.is_simple is True. " f"Set is_simple = False on the class.",
            )
        if resolved.model_type == "simple" and not cls_is_simple:
            raise TypeError(
                f"Model '{resolved.model_id}' resolved as simple but " f"{cls.__name__}.is_simple is False. " f"Set is_simple = True or register a chat_class_path.",
            )

    def load_entrypoint_manifests(
        self,
        group: str = "lmms_eval.models",
        *,
        overwrite: bool = False,
    ) -> None:
        """Load model manifests from Python entry points.

        Supported payloads per entry point:
        - `ModelManifest`
        - `Iterable[ModelManifest]`
        - `Callable[[], ModelManifest | Iterable[ModelManifest]]`
        """

        selected = self._select_entry_points(group)
        for ep in selected:
            payload = ep.load()
            manifests = self._coerce_payload_to_manifests(payload)
            for manifest in manifests:
                self.register_manifest(manifest, overwrite=overwrite)

    def list_model_names(self) -> list[str]:
        """Return all known requestable model names (ids + aliases)."""

        return sorted(self._alias_to_model_id)

    def list_canonical_model_ids(self) -> list[str]:
        """Return canonical model ids."""

        return sorted(self._manifests)

    def get_manifest(self, model_name: str) -> ModelManifest:
        """Return canonical manifest for a model id or alias."""

        model_id = self._require_model_id(model_name)
        return self._manifests[model_id]

    def _require_model_id(self, model_name: str) -> str:
        model_id = self._alias_to_model_id.get(model_name)
        if model_id is not None:
            return model_id

        available = ", ".join(self.list_model_names())
        raise ValueError(
            f"Model '{model_name}' not found in available models: {available}",
        )

    def _merge_manifest(
        self,
        current: ModelManifest | None,
        incoming: ModelManifest,
        *,
        overwrite: bool,
    ) -> ModelManifest:
        if current is None:
            return incoming

        simple_class_path = self._merge_class_path(
            current.simple_class_path,
            incoming.simple_class_path,
            overwrite=overwrite,
            field_name="simple_class_path",
            model_id=incoming.model_id,
        )
        chat_class_path = self._merge_class_path(
            current.chat_class_path,
            incoming.chat_class_path,
            overwrite=overwrite,
            field_name="chat_class_path",
            model_id=incoming.model_id,
        )

        aliases = tuple(dict.fromkeys((*current.aliases, *incoming.aliases)))

        return ModelManifest(
            model_id=incoming.model_id,
            simple_class_path=simple_class_path,
            chat_class_path=chat_class_path,
            aliases=aliases,
        )

    def _merge_class_path(
        self,
        current_value: str | None,
        incoming_value: str | None,
        *,
        overwrite: bool,
        field_name: str,
        model_id: str,
    ) -> str | None:
        if incoming_value is None:
            return current_value
        if current_value is None:
            return incoming_value
        if current_value == incoming_value:
            return current_value
        if overwrite:
            return incoming_value
        raise ValueError(
            f"Conflicting {field_name} for model '{model_id}': " f"'{current_value}' vs '{incoming_value}'",
        )

    def _coerce_payload_to_manifests(self, payload) -> list[ModelManifest]:
        if callable(payload):
            payload = payload()

        if isinstance(payload, ModelManifest):
            return [payload]

        if isinstance(payload, Iterable) and not isinstance(payload, (str, bytes, dict)):
            return self._coerce_manifest_iterable(payload)

        raise TypeError(
            "Entry point payload must be ModelManifest, iterable of ModelManifest, " "or callable returning one of those",
        )

    @staticmethod
    def _coerce_manifest_iterable(payload: Iterable[object]) -> list[ModelManifest]:
        manifests: list[ModelManifest] = []
        for item in payload:
            if not isinstance(item, ModelManifest):
                raise TypeError(
                    "Entry point iterable must contain only ModelManifest objects",
                )
            manifests.append(item)
        return manifests

    def _select_entry_points(self, group: str):
        eps = entry_points()
        if hasattr(eps, "select"):
            return list(eps.select(group=group))

        if isinstance(eps, dict):
            legacy_group = eps.get(group, [])
            if isinstance(legacy_group, Iterable):
                return list(legacy_group)
        return []
