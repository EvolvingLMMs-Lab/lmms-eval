"""Unified verification module for lmms-eval.

Provides a two-layer abstraction:

* **Extractors** — per-sample text transforms (strip reasoning, regex, MCQ, …)
* **Verifiers**  — correctness judges (rule-based, OpenAI, Gemini, composite)

Both are composed via :class:`VerificationPipeline`, which any task can
plug into its ``process_results``.

Quick-start::

    from lmms_eval.verifiers import build_verification_pipeline

    pipeline = build_verification_pipeline({
        "extractors": [{"type": "strip_reasoning"}],
        "verifier":   {"type": "openai", "model": "gpt-4o-2024-11-20"},
    })
    result = pipeline(question, prediction, ground_truth)
"""

from typing import Any, Dict, Type

from .base import Extractor, Verifier, VerifyResult, parse_binary_response
from .composite import CompositeVerifier
from .extractors import (
    MCQExtractor,
    NumberExtractor,
    RegexExtractor,
    StripExtractor,
    StripReasoningExtractor,
)
from .pipeline import VerificationPipeline
from .rule import (
    ContainsVerifier,
    ExactMatchVerifier,
    MCQMatchVerifier,
    NumericToleranceVerifier,
)

# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

EXTRACTOR_REGISTRY: Dict[str, Type[Extractor]] = {
    "strip_reasoning": StripReasoningExtractor,
    "regex": RegexExtractor,
    "mcq": MCQExtractor,
    "number": NumberExtractor,
    "strip": StripExtractor,
}

VERIFIER_REGISTRY: Dict[str, Type[Verifier]] = {
    "exact_match": ExactMatchVerifier,
    "contains": ContainsVerifier,
    "mcq_match": MCQMatchVerifier,
    "numeric": NumericToleranceVerifier,
    "composite": CompositeVerifier,
}

# Lazy-registered providers (avoid hard import of openai / google-genai)
_LAZY_VERIFIERS = {
    "openai": ("lmms_eval.verifiers.openai", "OpenAIVerifier"),
    "gemini": ("lmms_eval.verifiers.gemini", "GeminiVerifier"),
}


def _resolve_verifier(name: str) -> Type[Verifier]:
    if name in VERIFIER_REGISTRY:
        return VERIFIER_REGISTRY[name]
    if name in _LAZY_VERIFIERS:
        module_path, cls_name = _LAZY_VERIFIERS[name]
        import importlib

        mod = importlib.import_module(module_path)
        cls = getattr(mod, cls_name)
        VERIFIER_REGISTRY[name] = cls
        return cls
    raise KeyError(f"Unknown verifier type: {name!r}.  Available: {sorted(set(VERIFIER_REGISTRY) | set(_LAZY_VERIFIERS))}")


def _resolve_extractor(name: str) -> Type[Extractor]:
    if name in EXTRACTOR_REGISTRY:
        return EXTRACTOR_REGISTRY[name]
    raise KeyError(f"Unknown extractor type: {name!r}.  Available: {sorted(EXTRACTOR_REGISTRY)}")


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_verification_pipeline(config: Dict[str, Any]) -> VerificationPipeline:
    """Build a :class:`VerificationPipeline` from a config dict.

    Config schema::

        {
            "extractors": [
                {"type": "strip_reasoning"},
                {"type": "regex", "pattern": r"Answer:\\s*(.+)"},
            ],
            "verifier": {
                "type": "openai",
                "model": "gpt-4o-2024-11-20",
                "judge_type": "binary",
            },
        }
    """
    extractors = []
    for ext_cfg in config.get("extractors", []):
        ext_cfg = dict(ext_cfg)  # copy to avoid mutation
        ext_type = ext_cfg.pop("type")
        ext_cls = _resolve_extractor(ext_type)
        extractors.append(ext_cls(**ext_cfg))

    ver_cfg = dict(config.get("verifier", {}))
    ver_type = ver_cfg.pop("type")

    # Handle nested composite verifiers
    if ver_type == "composite":
        sub_cfgs = ver_cfg.pop("verifiers", [])
        sub_verifiers = []
        for sub_cfg in sub_cfgs:
            sub_cfg = dict(sub_cfg)
            sub_type = sub_cfg.pop("type")
            sub_cls = _resolve_verifier(sub_type)
            sub_verifiers.append(sub_cls(**sub_cfg))
        verifier = CompositeVerifier(verifiers=sub_verifiers)
    else:
        ver_cls = _resolve_verifier(ver_type)
        verifier = ver_cls(**ver_cfg)

    return VerificationPipeline(extractors=extractors, verifier=verifier)


__all__ = [
    # Core types
    "Extractor",
    "Verifier",
    "VerifyResult",
    "parse_binary_response",
    # Pipeline
    "VerificationPipeline",
    "build_verification_pipeline",
    # Extractors
    "StripReasoningExtractor",
    "RegexExtractor",
    "MCQExtractor",
    "NumberExtractor",
    "StripExtractor",
    # Verifiers
    "ExactMatchVerifier",
    "ContainsVerifier",
    "MCQMatchVerifier",
    "NumericToleranceVerifier",
    "CompositeVerifier",
    # Registries
    "EXTRACTOR_REGISTRY",
    "VERIFIER_REGISTRY",
]
