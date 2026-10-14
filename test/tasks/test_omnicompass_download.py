"""Offline download/extraction contract tests; never fetch real media archives."""

import hashlib
import io
import json
import os
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import pytest

from lmms_eval.tasks.omnicompass._core import hf_data
from lmms_eval.tasks.omnicompass._core.hf_data import METADATA, DatasetStore


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def repository(tmp_path, monkeypatch):
    remote = tmp_path / "remote"
    remote.mkdir()
    contents = {"videos/oc_001.mp4": b"video1", "videos/oc_002.mp4": b"video2", "references/oc_001.png": b"reference"}
    manifest, archives = [], []
    for name, data in contents.items():
        manifest.append({"public_path": name, "bytes": len(data), "sha256": digest(data)})
    for kind in ("videos", "references"):
        names = sorted(n for n in contents if n.startswith(kind + "/"))
        path = remote / "media" / (kind + "-000.tar")
        path.parent.mkdir(exist_ok=True)
        with tarfile.open(path, "w") as archive:
            for name in names:
                item = tarfile.TarInfo(name)
                item.size = len(contents[name])
                archive.addfile(item, io.BytesIO(contents[name]))
        archives.append({"path": "media/" + path.name, "bytes": path.stat().st_size, "sha256": digest(path.read_bytes()), "files": len(names), "first_path": names[0], "last_path": names[-1]})
    for name in METADATA:
        write_json(remote / name, {})
    (remote / "media_manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in manifest))
    write_json(remote / "archives.json", archives)
    release = {"schema_version": "1.1", "audio_delivery": "embedded_in_original_video", "files": {n: digest((remote / n).read_bytes()) for n in METADATA}}
    write_json(remote / "release.json", release)
    calls = []

    def download(repo_id, filename, *, repo_type, revision):
        assert repo_id == hf_data.REPO_ID and repo_type == "dataset"
        assert revision == hf_data.DEFAULT_REVISION
        calls.append(filename)
        return str(remote / filename)

    monkeypatch.setattr(hf_data, "hf_hub_download", download)
    monkeypatch.delenv("OMNICOMPASS_MAX_DOWNLOAD_BYTES", raising=False)
    return remote, tmp_path / "cache", calls


def test_metadata_only_and_revision_pin(repository):
    remote, cache, calls = repository
    store = DatasetStore(cache=cache)
    assert len(store.manifest) == 3
    assert set(calls) == {"release.json", *METADATA}
    calls.clear()
    DatasetStore(cache=cache)
    assert not calls


def test_lazy_shard_cache_and_references(repository):
    _, cache, calls = repository
    store = DatasetStore(cache=cache)
    calls.clear()
    store.ensure(["videos/oc_001.mp4"])
    assert calls == ["media/videos-000.tar"]
    store.ensure(["videos/oc_002.mp4", "videos/oc_001.mp4"])
    assert calls == ["media/videos-000.tar"]
    DatasetStore(cache=cache).ensure(["videos/oc_002.mp4"])
    assert calls == ["media/videos-000.tar"]
    store.ensure(["references/oc_001.png"])
    assert calls == ["media/videos-000.tar", "media/references-000.tar"]


def test_local_root_never_downloads(repository):
    _, cache, calls = repository
    remote_store = DatasetStore(cache=cache)
    calls.clear()
    local = DatasetStore(root=remote_store.root, cache=cache)
    with pytest.raises(FileNotFoundError):
        local.ensure(["videos/oc_001.mp4"])
    assert not calls


def test_budget_prevents_media_get(repository, monkeypatch):
    _, cache, calls = repository
    store = DatasetStore(cache=cache)
    calls.clear()
    monkeypatch.setenv("OMNICOMPASS_MAX_DOWNLOAD_BYTES", "0")
    with pytest.raises(ValueError, match="byte limit"):
        store.ensure(["videos/oc_001.mp4"])
    assert not calls


def test_bad_metadata_hash_rejected(repository):
    remote, cache, _ = repository
    (remote / "questions.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="Metadata checksum"):
        DatasetStore(cache=cache)


def test_cached_metadata_still_checked(repository):
    _, cache, _ = repository
    store = DatasetStore(cache=cache)
    (store.root / "questions.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="Metadata checksum"):
        DatasetStore(cache=cache)


def test_downloaded_archive_hash_checked(repository):
    remote, cache, _ = repository
    store = DatasetStore(cache=cache)
    archive = remote / "media/videos-000.tar"
    archive.write_bytes(b"x" * archive.stat().st_size)
    with pytest.raises(ValueError, match="Archive checksum"):
        store.ensure(["videos/oc_001.mp4"])
    assert not (store.root / "videos/oc_001.mp4").exists()


def test_existing_media_corruption_is_not_overwritten(repository):
    _, cache, _ = repository
    store = DatasetStore(cache=cache)
    store.ensure(["videos/oc_001.mp4"])
    video = store.root / "videos/oc_001.mp4"
    video.write_bytes(b"broken")
    with pytest.raises(ValueError, match="Media checksum"):
        store.ensure(["videos/oc_001.mp4"])
    assert video.read_bytes() == b"broken"


def test_same_size_corruption_with_restored_mtime(repository):
    _, cache, _ = repository
    store = DatasetStore(cache=cache)
    store.ensure(["videos/oc_001.mp4"])
    video = store.root / "videos/oc_001.mp4"
    stat = video.stat()
    video.write_bytes(b"broken")
    os.utime(video, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match="Media checksum"):
        store.ensure(["videos/oc_001.mp4"])


def test_corruption_with_all_stat_fields_unchanged(repository, monkeypatch):
    _, cache, calls = repository
    store = DatasetStore(cache=cache)
    store.ensure(["videos/oc_001.mp4"])
    video = store.root / "videos/oc_001.mp4"
    before = video.stat()
    video.write_bytes(b"broken")
    original_stat = Path.stat
    monkeypatch.setattr(
        Path,
        "stat",
        lambda path, *a, **kw: before if path == video else original_stat(path, *a, **kw),
    )
    calls.clear()
    with pytest.raises(ValueError, match="Media checksum"):
        store.ensure(["videos/oc_001.mp4"])
    assert not calls
    assert video.read_bytes() == b"broken"


def test_extracted_content_hash_checked(repository):
    _, cache, _ = repository
    store = DatasetStore(cache=cache)
    store.manifest["videos/oc_001.mp4"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="Extracted checksum"):
        store.ensure(["videos/oc_001.mp4"])
    assert not (store.root / "videos/oc_001.mp4").exists()


@pytest.mark.parametrize("mode", ["traversal", "symlink", "duplicate", "missing", "bad_size"])
def test_invalid_archive_members_rejected(repository, mode):
    remote, cache, _ = repository
    store = DatasetStore(cache=cache)
    archive = remote / "media/videos-000.tar"
    with tarfile.open(archive, "w") as tar:
        names = ["videos/oc_001.mp4", "videos/oc_002.mp4"]
        if mode == "traversal":
            names[0] = "../outside"
        if mode == "duplicate":
            names[1] = names[0]
        if mode == "missing":
            names = names[:1]
        for name in names:
            item = tarfile.TarInfo(name)
            data = b"video1" if name.endswith("001.mp4") else b"video2"
            if mode == "bad_size":
                data += b"x"
            item.size = len(data)
            if mode == "symlink":
                item.type = tarfile.SYMTYPE
                item.linkname = "../outside"
                item.size = 0
            tar.addfile(item, io.BytesIO(data))
    info = store.archive_for["videos/oc_001.mp4"]
    info.update(bytes=archive.stat().st_size, sha256=digest(archive.read_bytes()))
    with pytest.raises(ValueError):
        store.ensure(["videos/oc_001.mp4"])
    assert not (store.root.parent / "outside").exists()


def test_symlink_destination_rejected(repository, tmp_path):
    _, cache, _ = repository
    store = DatasetStore(cache=cache)
    outside = tmp_path / "outside"
    outside.mkdir()
    (store.root / "videos").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="Unsafe"):
        store.ensure(["videos/oc_001.mp4"])
    assert list(outside.iterdir()) == []


def test_failed_download_can_resume(repository, monkeypatch):
    _, cache, calls = repository
    store = DatasetStore(cache=cache)
    original = hf_data.hf_hub_download
    monkeypatch.setattr(hf_data, "hf_hub_download", Mock(side_effect=ConnectionError("interrupted")))
    with pytest.raises(ConnectionError):
        store.ensure(["videos/oc_001.mp4"])
    monkeypatch.setattr(hf_data, "hf_hub_download", original)
    calls.clear()
    store.ensure(["videos/oc_001.mp4"])
    assert calls == ["media/videos-000.tar"]


def test_concurrent_preparation_downloads_once(repository):
    _, cache, calls = repository
    stores = [DatasetStore(cache=cache), DatasetStore(cache=cache)]
    calls.clear()
    with ThreadPoolExecutor(2) as pool:
        list(pool.map(lambda store: store.ensure(["videos/oc_001.mp4"]), stores))
    assert calls == ["media/videos-000.tar"]


def test_unknown_media_rejected(repository):
    _, cache, calls = repository
    store = DatasetStore(cache=cache)
    calls.clear()
    with pytest.raises(ValueError, match="Unlisted"):
        store.ensure(["videos/oc_999.mp4"])
    assert not calls
