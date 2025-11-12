# samplers/__init__.py
from __future__ import annotations

from typing import Dict, Type, Any, Optional, Iterable
import importlib
import pkgutil

# ---- Registry --------------------------------------------------------------

VIDEO_SAMPLER_REGISTRY: Dict[str, Type] = {}

def register_video_sampler(name: str, *, overwrite: bool = False):
    """Class decorator to register a sampler class under a name."""
    def decorate(cls: Type):
        if not overwrite and name in VIDEO_SAMPLER_REGISTRY:
            raise KeyError(
                f"Sampler name {name!r} already registered with "
                f"{VIDEO_SAMPLER_REGISTRY[name]!r}"
            )
        from .base import BaseVideoSampler
        if not issubclass(cls, BaseVideoSampler):
            raise TypeError(f"{cls.__name__} must subclass BaseVideoSampler")
        VIDEO_SAMPLER_REGISTRY[name] = cls
        return cls
    return decorate

# ---- Lookups ---------------------------------------------------------------

def get_video_sampler_cls(name: str) -> Type:
    """Return the registered class for a sampler `name`."""
    try:
        return VIDEO_SAMPLER_REGISTRY[name]
    except KeyError as e:
        known = ", ".join(sorted(VIDEO_SAMPLER_REGISTRY)) or "<none>"
        raise KeyError(f"Unknown video sampler {name!r}. Known: {known}") from e

# ---- Optional: auto-discover submodules so decorators run ------------------

def _auto_import_submodules(package_name: str, exclude: Iterable[str] = ()):
    """Import all submodules of this package so @register_... executes.

    Call once at import time. Skips names in `exclude`.
    """
    pkg = importlib.import_module(package_name)
    if not hasattr(pkg, "__path__"):  # not a package
        return
    for m in pkgutil.iter_modules(pkg.__path__):
        mod_name = f"{package_name}.{m.name}"
        if m.name in exclude:
            continue
        importlib.import_module(mod_name)

# Import all samplers on package import (edit excludes as needed)
_auto_import_submodules(__name__, exclude=("base", "__init__"))

__all__ = [
    "register_video_sampler",
    "get_video_sampler",
    "get_video_sampler_cls",
    "VIDEO_SAMPLER_REGISTRY",
]
