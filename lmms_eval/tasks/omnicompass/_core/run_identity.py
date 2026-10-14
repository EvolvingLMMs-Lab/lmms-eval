"""Cache identity for model artifacts and inference dependencies."""

import os
import platform
from importlib import metadata
from pathlib import Path
from typing import Any

from .release_tools import sha


def runtime_identity() -> dict[str, Any]:
    """Record dependency versions that can affect model inputs and outputs."""
    packages = (
        "torch",
        "transformers",
        "accelerate",
        "qwen-vl-utils",
        "qwen-omni-utils",
        "flash-attn",
        "vllm",
        "lmdeploy",
        "av",
        "Pillow",
        "openai",
        "requests",
    )
    versions = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "packages": versions}


def model_identity(model_path: str | os.PathLike | None = None, revision: str | None = None) -> dict[str, Any]:
    """Hash local artifacts once at run entry; remote revisions are declarations."""
    result = {"declared_revision": revision or os.getenv("OMNICOMPASS_MODEL_REVISION") or None}
    if not isinstance(model_path, (str, os.PathLike)):
        return result
    root = Path(model_path).expanduser()
    if not root.exists():
        return result
    candidates = [root] if root.is_file() else sorted(root.rglob("*"))
    suffixes = {".safetensors", ".bin", ".pt", ".pth", ".gguf", ".json", ".model", ".jinja"}
    tokenizer_files = {"merges.txt", "vocab.txt", "tokenizer.tiktoken"}
    artifacts = {}
    for path in candidates:
        if not path.is_file() or (path.suffix not in suffixes and path.name not in tokenizer_files):
            continue
        artifacts[path.name if root.is_file() else str(path.relative_to(root))] = sha(path)
    if not artifacts:
        raise ValueError("Local model path contains no recognized model/tokenizer artifacts")
    result["local_artifacts_sha256"] = artifacts
    return result
