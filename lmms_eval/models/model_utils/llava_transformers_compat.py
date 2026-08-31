"""Compatibility shim for LLaVA + transformers version mismatch.

External LLaVA-NeXT code historically imported

    from transformers.modeling_utils import apply_chunking_to_forward

whereas transformers >= 4.47 moved those helpers to

    transformers.pytorch_utils.

Newer LLaVA patches swap the import, but older installs still
reference the old location, and some forks mistakenly import
``PreTrainedModel`` from ``pytorch_utils``.  This module mirrors the
helpers to both locations so that either import path succeeds
regardless of the installed transformers version, without pinning
the dependency.
"""

from __future__ import annotations


def ensure_transformers_compat() -> None:
    try:
        import transformers.modeling_utils as modeling_utils
    except Exception:
        modeling_utils = None  # type: ignore[assignment]
    try:
        import transformers.pytorch_utils as pytorch_utils
    except Exception:
        pytorch_utils = None  # type: ignore[assignment]

    if modeling_utils is None and pytorch_utils is None:
        return

    names = [
        "apply_chunking_to_forward",
        "find_pruneable_heads_and_indices",
        "prune_linear_layer",
    ]
    for name in names:
        mu_has = hasattr(modeling_utils, name) if modeling_utils is not None else False
        pu_has = hasattr(pytorch_utils, name) if pytorch_utils is not None else False
        if pu_has and not mu_has:
            try:
                setattr(modeling_utils, name, getattr(pytorch_utils, name))
            except Exception:
                pass
        if mu_has and not pu_has:
            try:
                setattr(pytorch_utils, name, getattr(modeling_utils, name))
            except Exception:
                pass

    # Some patched LLaVA forks incorrectly import PreTrainedModel from
    # pytorch_utils; ensure that path also resolves.
    if modeling_utils is not None and pytorch_utils is not None:
        if hasattr(modeling_utils, "PreTrainedModel") and not hasattr(pytorch_utils, "PreTrainedModel"):
            try:
                setattr(pytorch_utils, "PreTrainedModel", getattr(modeling_utils, "PreTrainedModel"))
            except Exception:
                pass
