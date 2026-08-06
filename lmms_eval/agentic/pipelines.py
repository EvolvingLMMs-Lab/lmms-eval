"""Selection and execution of task-local, model-specific parser pipelines."""

from __future__ import annotations

from fnmatch import fnmatchcase
from typing import Any

from lmms_eval.agentic.types import ParserContext


def select_parser_pipelines(model_specific_parsers: Any, model_name: str | None) -> dict[str, Any]:
    """Select parser pipelines for ``model_name`` and merge them over default.

    ``model_specific_parsers`` is declared by a task, for example::

        {
            "default": {"observation": observation_parser, "action": action_parser},
            "*Qwen*": {"action": [qwen_parser, action_parser]},
        }

    Exact model keys win over glob patterns.  Glob matching is
    case-insensitive and otherwise follows :mod:`fnmatch`.  A model-specific
    entry may override only one stage; omitted stages inherit from ``default``.
    """

    if not isinstance(model_specific_parsers, dict):
        raise TypeError("model_specific_parsers must be a mapping from model patterns to parser pipelines")

    default = _pipeline_mapping(model_specific_parsers.get("default", {}), pattern="default")
    selected: dict[str, Any] = {}
    if model_name:
        normalized_name = str(model_name).casefold()
        exact_pattern = next(
            (pattern for pattern in model_specific_parsers if pattern != "default" and str(pattern).casefold() == normalized_name),
            None,
        )
        matched_pattern = exact_pattern
        if matched_pattern is None:
            matched_pattern = next(
                (pattern for pattern in model_specific_parsers if pattern != "default" and fnmatchcase(normalized_name, str(pattern).casefold())),
                None,
            )
        if matched_pattern is not None:
            selected = _pipeline_mapping(model_specific_parsers[matched_pattern], pattern=str(matched_pattern))

    pipelines = {**default, **selected}
    missing = [stage for stage in ("observation", "action") if stage not in pipelines]
    if missing:
        raise ValueError(f"model_specific_parsers has no {', '.join(missing)} pipeline for model {model_name!r}")
    return pipelines


def apply_parser_pipeline(value: Any, pipeline: Any, context: ParserContext) -> Any:
    """Apply one task-owned ``Any -> Any`` function, or a sequence of them."""

    parsers = pipeline if isinstance(pipeline, (list, tuple)) else [pipeline]
    current = value
    for parser in parsers:
        if not callable(parser):
            raise TypeError(f"parser pipeline entries must be callable, got {type(parser).__name__}")
        current = parser(current, context)
    return current


def _pipeline_mapping(value: Any, *, pattern: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"model_specific_parsers[{pattern!r}] must be a mapping of pipeline stages")
    return dict(value)
