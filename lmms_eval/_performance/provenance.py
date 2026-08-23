from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path

from .json_v1 import canonical_json_bytes

try:
    import resource as _resource
except ImportError:
    _resource = None

_SOURCE_DIGEST_DOMAIN = b"lmms-eval/SourceTreeV1/manifest"


@dataclass(frozen=True)
class BaselineProvenance:
    source_commit: str
    source_tree_digest: str
    environment_lock_digest: str
    hardware: str


def _source_revision(repo_root: Path) -> tuple[str, str]:
    resolved_repo_root = repo_root.resolve(strict=True)
    git_toplevel = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=resolved_repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if resolved_repo_root != Path(git_toplevel).resolve(strict=True):
        raise ValueError("repo_root must be the Git repository root")
    commit = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=resolved_repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not commit:
        raise RuntimeError("Git commit is missing")
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", "lmms_eval", "tools/benchmark_eval_phases.py", "pyproject.toml"],
        cwd=resolved_repo_root,
        check=True,
        capture_output=True,
        text=False,
    ).stdout
    paths = sorted(path for path in listed.decode("utf-8").split("\0") if path)
    manifest = []
    for path in paths:
        resolved_file = (resolved_repo_root / path).resolve(strict=True)
        try:
            resolved_file.relative_to(resolved_repo_root)
        except ValueError as exc:
            raise ValueError(f"Source path escapes repository root: {path}") from exc
        manifest.append({"path": path, "sha256": hashlib.sha256(resolved_file.read_bytes()).hexdigest()})
    digest = hashlib.sha256(_SOURCE_DIGEST_DOMAIN + b"\0" + canonical_json_bytes(manifest)).hexdigest()
    return commit, f"sha256:{digest}"


def _environment_lock_digest() -> str:
    rows = sorted(f"{distribution.metadata['Name']}=={distribution.version}".casefold() for distribution in importlib_metadata.distributions())
    digest = hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _hardware_description() -> str:
    return f"platform={platform.platform()}; machine={platform.machine()}; cpu_count={os.cpu_count()}"


def capture_baseline_provenance(repo_root: Path) -> BaselineProvenance:
    source_commit, source_tree_digest = _source_revision(repo_root)
    return BaselineProvenance(
        source_commit=source_commit,
        source_tree_digest=source_tree_digest,
        environment_lock_digest=_environment_lock_digest(),
        hardware=_hardware_description(),
    )


def peak_host_rss_bytes() -> int | None:
    if _resource is None:
        return None
    rss = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss * 1024)
