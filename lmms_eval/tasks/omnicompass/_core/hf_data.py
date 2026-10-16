"""Pinned Hugging Face metadata and lazy, verified media preparation."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tarfile
import tempfile
from pathlib import Path

import portalocker
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.constants import HF_HOME

from .release_tools import rows, safe, sha

REPO_ID = "VCLab-PolyU/OmniCompass"
DEFAULT_REVISION = "2b82d4e8a2b57c14bd887e06d96a5bb5534e7b7a"
METADATA = (
    "videos.jsonl",
    "questions.jsonl",
    "schema.json",
    "taxonomy.json",
    "archives.json",
    "media_manifest.jsonl",
)
logger = logging.getLogger(__name__)


def default_cache():
    return Path(os.getenv("OMNICOMPASS_CACHE", str(Path(HF_HOME) / "omnicompass"))).expanduser()


def atomic_copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as dst:
            temporary = Path(dst.name)
            with Path(source).open("rb") as src:
                shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def validate_metadata(root):
    release = json.loads((root / "release.json").read_text())
    if release.get("schema_version") != "1.1":
        raise ValueError("Unsupported OmniCompass release schema")
    if release.get("audio_delivery") != "embedded_in_original_video":
        raise ValueError("Expected original MP4 media with embedded audio")
    files = release["files"]
    for name in METADATA:
        if name not in files or sha(safe(root, name)) != files[name]:
            raise ValueError(f"Metadata checksum mismatch: {name}")
    return release


class DatasetStore:
    """An explicit local root is read-only; otherwise use a revision-scoped cache."""

    def __init__(self, root=None, cache=None, revision=None):
        self.cache = Path(cache or default_cache()).expanduser().resolve()
        self.local = root is not None
        self.revision = revision or os.getenv("OMNICOMPASS_REVISION", DEFAULT_REVISION)
        if self.local:
            self.root = Path(root).expanduser().resolve(strict=True)
        else:
            if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
                self.revision = HfApi().dataset_info(REPO_ID, revision=self.revision).sha
            self.root = self.cache / "releases" / self.revision
            self.root.mkdir(parents=True, exist_ok=True)
            with portalocker.Lock(str(self.root / ".metadata.lock"), timeout=600):
                if not (self.root / ".metadata-complete").exists():
                    release_file = self._download("release.json")
                    release = json.loads(release_file.read_text())
                    if not set(METADATA) <= release.get("files", {}).keys():
                        raise ValueError("Release does not contain the required metadata")
                    for name in METADATA:
                        source = self._download(name)
                        if sha(source) != release["files"][name]:
                            raise ValueError(f"Metadata checksum mismatch: {name}")
                        atomic_copy(source, safe(self.root, name))
                    atomic_copy(release_file, self.root / "release.json")
                    validate_metadata(self.root)
                    (self.root / ".metadata-complete").touch()
        self.release = validate_metadata(self.root)
        manifest = rows(self.root / "media_manifest.jsonl")
        self.manifest = {r["public_path"]: r for r in manifest}
        if len(manifest) != len(self.manifest):
            raise ValueError("Duplicate media manifest paths")
        self.archives = json.loads((self.root / "archives.json").read_text())
        self.archive_for = {}
        for name in self.manifest:
            safe(self.root, name)
            matches = [a for a in self.archives if a["first_path"] <= name <= a["last_path"]]
            if len(matches) != 1:
                raise ValueError(f"Media must belong to exactly one archive: {name}")
            self.archive_for[name] = matches[0]

    def _download(self, name):
        return Path(hf_hub_download(REPO_ID, name, repo_type="dataset", revision=self.revision))

    def _valid(self, name):
        expected = self.manifest[name]
        path = safe(self.root, name)
        if not path.exists():
            return False
        stat = path.stat()
        # Network filesystems can retain the same timestamps after a write.
        # Verify content even when every stat field matches a previous check.
        if not path.is_file() or stat.st_size != expected["bytes"] or sha(path) != expected["sha256"]:
            if self.local or not path.is_file():
                raise ValueError(f"Media checksum mismatch: {name}")
            # Managed cache files can be replaced atomically from a verified archive.
            # Keep the old file until extraction succeeds; explicit local roots are read-only.
            return False
        return True

    def ensure(self, names):
        for name in dict.fromkeys(names):
            if name not in self.manifest:
                raise ValueError(f"Unlisted media: {name}")
            if self._valid(name):
                continue
            if self.local:
                raise FileNotFoundError(f"Missing media in OMNICOMPASS_DATA: {name}")
            info = self.archive_for[name]
            safe(self.root, info["path"])
            lock_dir = self.cache / "download_locks" / self.revision
            lock_dir.mkdir(parents=True, exist_ok=True)
            with portalocker.Lock(str(lock_dir / (Path(info["path"]).name + ".lock")), timeout=86400):
                if self._valid(name):
                    continue
                limit = os.getenv("OMNICOMPASS_MAX_DOWNLOAD_BYTES")
                if limit is not None and info["bytes"] > int(limit):
                    raise ValueError(f"Media download exceeds configured byte limit: {info['path']}")
                logger.info("Preparing %s (%s bytes); subsequent runs reuse the cache", info["path"], info["bytes"])
                archive = self._download(info["path"])
                self._extract(archive, info)
                if not self._valid(name):
                    raise ValueError(f"Archive did not supply {name}")

    def _extract(self, source, info):
        if source.stat().st_size != info["bytes"] or sha(source) != info["sha256"]:
            raise ValueError(f"Archive checksum mismatch: {info['path']}")
        expected_names = {name for name in self.manifest if info["first_path"] <= name <= info["last_path"]}
        seen = set()
        with tarfile.open(source, "r:") as archive:
            for member in archive:
                if not member.isfile() or member.name not in expected_names or member.name in seen:
                    raise ValueError(f"Unexpected archive member: {member.name}")
                seen.add(member.name)
                expected = self.manifest[member.name]
                if member.size != expected["bytes"]:
                    raise ValueError(f"Member size mismatch: {member.name}")
                if self._valid(member.name):
                    continue
                target = safe(self.root, member.name)
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as dst:
                        temporary = Path(dst.name)
                        with archive.extractfile(member) as src:
                            shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
                    if sha(temporary) != expected["sha256"]:
                        raise ValueError(f"Extracted checksum mismatch: {member.name}")
                    temporary.replace(target)
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
        if seen != expected_names or len(seen) != info["files"]:
            raise ValueError("Incomplete archive member list")
