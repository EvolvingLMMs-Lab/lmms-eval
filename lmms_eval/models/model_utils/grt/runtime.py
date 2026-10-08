"""Compatibility guard for model-internal GRT patching."""

from importlib.metadata import version

from lmms_eval.imports import require_package


def require_grt_runtime() -> None:
    """Reject unvalidated Transformers internals before loading model weights."""
    actual = version("transformers")
    if actual != "4.57.6":
        raise RuntimeError(f"GRT's published patch kernels require transformers==4.57.6, found {actual}. Install the grt extra in an isolated environment.")
    require_package("qwen_vl_utils", extras="grt", feature="GRT profiles")
