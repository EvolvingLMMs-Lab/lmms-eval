"""Spec resolution for agentic components.

A component spec can be:

- an instance of the expected type (returned as-is),
- a callable factory (called with signature-filtered kwargs),
- a registry short name, e.g. ``"openai"``,
- an import path ``"my_pkg.servers:CustomServer"`` (or dotted form),
- a dict ``{"name": <short name or import path>, **kwargs}``.

Registry values are import-path strings so that importing
``lmms_eval.agentic`` never pulls optional heavy dependencies.
"""

from __future__ import annotations

import inspect
from importlib import import_module
from typing import Any

REGISTRY: dict[str, dict[str, str]] = {
    "model_server": {
        "openai": "lmms_eval.agentic.servers:OpenAIModelServer",
        "debug": "lmms_eval.agentic.servers:FixedActionModelServer",
    },
    "model_output_parser": {
        "identity": "lmms_eval.agentic.parsers:IdentityModelOutputParser",
        "qwen": "lmms_eval.agentic.parsers:QwenModelOutputParser",
    },
    "observation_parser": {},
    "action_parser": {
        "action_name": "lmms_eval.agentic.parsers:ActionNameParser",
    },
}


def resolve(kind: str, spec: Any, *, expected: type, **context_kwargs: Any) -> Any:
    """Build the ``kind`` component described by ``spec``.

    ``context_kwargs`` (e.g. ``doc``, ``lmms_eval_specific_kwargs``) are offered
    to the factory and filtered by its signature; explicit kwargs from a dict
    spec win over context.
    """

    if spec is None:
        raise TypeError(f"{kind} spec is required")
    if isinstance(spec, expected):
        return spec

    spec_kwargs: dict[str, Any] = {}
    factory = spec
    if isinstance(spec, dict):
        spec_kwargs = dict(spec)
        factory = spec_kwargs.pop("name", None) or spec_kwargs.pop("factory", None)
        if factory is None:
            raise TypeError(f"{kind} dict spec requires a 'name' or 'factory' field")
    factory = _resolve_factory(kind, factory)

    component = call_with_accepted_kwargs(factory, {**context_kwargs, **spec_kwargs})
    if not isinstance(component, expected):
        raise TypeError(f"{kind} factory returned {type(component).__name__}, expected {expected.__name__}")
    return component


def _resolve_factory(kind: str, factory: Any) -> Any:
    if not isinstance(factory, str):
        if callable(factory):
            return factory
        raise TypeError(f"Expected {kind} instance, callable, import path, or known name; got {type(factory).__name__}")
    aliases = REGISTRY.get(kind, {})
    if factory in aliases:
        return import_from_path(aliases[factory])
    if ":" in factory or "." in factory:
        return import_from_path(factory)
    available = ", ".join(sorted(aliases)) or "<none>"
    raise KeyError(f"Unknown {kind} '{factory}'. Available built-ins: {available}. You can also pass an import path.")


def import_from_path(path: str) -> Any:
    """Import ``module:attribute`` (preferred) or ``module.attribute``."""

    module_name, sep, attr = path.partition(":")
    if not sep:
        module_name, sep, attr = path.rpartition(".")
    if not module_name or not attr:
        raise ValueError(f"Import path must be 'module:attribute' or 'module.attribute', got {path!r}")
    return getattr(import_module(module_name), attr)


def call_with_accepted_kwargs(factory: Any, kwargs: dict[str, Any]) -> Any:
    """Call ``factory`` with only the kwargs its signature accepts."""

    signature = inspect.signature(factory)
    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return factory(**kwargs)
    return factory(**{name: value for name, value in kwargs.items() if name in parameters})
