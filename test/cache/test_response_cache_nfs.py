"""Opt-in NFS integration: LMMS_CACHE_E2E_ROOT=/mnt/nfs/tests pytest this_file.

Uses real mount detection and multiple processes, without mocking the filesystem.
A loopback NFS mount exercises the protocol, but not multi-host cache coherence.
"""

import multiprocessing
import os
import sqlite3
import tempfile
import time
import urllib.parse
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from lmms_eval.caching import response_cache as module
from lmms_eval.caching.fs_detect import FsType, detect_fs_type
from lmms_eval.caching.response_cache import ResponseCache


@contextmanager
def _local_sqlite_only():
    connect = sqlite3.connect

    def checked(database, *args, **kwargs):
        path = str(database)
        if path.startswith("file:"):
            path = urllib.parse.unquote(urllib.parse.urlsplit(path).path)
        assert detect_fs_type(path) == FsType.LOCAL, f"SQLite opened non-local path: {path}"
        return connect(database, *args, **kwargs)

    with patch.object(module.sqlite3, "connect", side_effect=checked):
        yield


@pytest.fixture
def nfs_root(monkeypatch):
    root = os.environ.get("LMMS_CACHE_E2E_ROOT")
    if not root or detect_fs_type(root) != FsType.REMOTE:
        pytest.skip("LMMS_CACHE_E2E_ROOT must point to a real remote filesystem")
    os.makedirs(root, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cache-nfs-", dir=root) as cache_root, tempfile.TemporaryDirectory(prefix="cache-local-") as scratch:
        assert detect_fs_type(scratch) == FsType.LOCAL
        monkeypatch.setenv("LMMS_LOCAL_CACHE", scratch)
        with _local_sqlite_only():
            yield cache_root
        for pattern in ("*-wal", "*-shm", "*-journal"):
            assert not list(Path(cache_root).rglob(pattern))


def _cache(root, run):
    os.environ["LMMS_CACHE_RUN_ID"] = run
    cache = ResponseCache.create(root, model="nfs-integration")
    assert cache._use_scratch
    assert detect_fs_type(cache.db_path) == FsType.LOCAL
    return cache


def _publish(root, run, count=1, barrier=None):
    with _local_sqlite_only():
        cache = _cache(root, run)
        try:
            for index in range(count):
                key = f"{run}:{index}"
                cache._store(key, "generate_until", "nfs_test", index, 0, {}, key)
            cache._checkpoint_to_run_dir()
            if barrier is not None:
                barrier.wait(timeout=90)
            cache.finalize(success=True)
        finally:
            cache.close()


def _assert_snapshot(root, expected):
    cache = _cache(root, "inspect")
    try:
        assert cache.get_stats()["shared_cached_entries"] == expected
        assert cache._shared_db.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert cache._shared_db.execute("PRAGMA journal_mode").fetchone() == ("delete",)
    finally:
        cache.close()


def test_nfs_parallel_publication_with_pinned_reader(nfs_root):
    _publish(nfs_root, "seed")
    reader = _cache(nfs_root, "pinned-reader")
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(4)
    workers = [context.Process(target=_publish, args=(nfs_root, f"worker-{index}", 16, barrier)) for index in range(4)]
    try:
        for worker in workers:
            worker.start()
        deadline = time.monotonic() + 120
        while any(worker.is_alive() for worker in workers):
            assert time.monotonic() < deadline, "NFS publishers did not finish"
            assert reader._lookup("seed:0") == "seed:0"
            assert reader._lookup("worker-0:0") is None
            time.sleep(0.05)
        for worker in workers:
            worker.join()
            assert worker.exitcode == 0
        assert reader.get_stats()["shared_cached_entries"] == 1
        _assert_snapshot(nfs_root, 65)
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            if worker.pid is not None:
                worker.join(timeout=10)
        reader.close()


def test_nfs_publication_failure_is_recovered_by_next_run(nfs_root):
    _publish(nfs_root, "seed")
    replace = os.replace

    def fail_root(source, destination):
        if destination == os.path.join(nfs_root, "cache.db"):
            raise OSError("injected publication failure")
        return replace(source, destination)

    with patch.object(module.os, "replace", side_effect=fail_root):
        with pytest.raises(OSError, match="injected publication"):
            _publish(nfs_root, "deferred")
    _assert_snapshot(nfs_root, 1)
    assert Path(nfs_root, "runs/deferred/.ready").exists()
    assert not Path(nfs_root, "runs/deferred/.merged").exists()
    assert not list(Path(nfs_root).rglob(".cache-*.tmp"))
    _publish(nfs_root, "next")
    _assert_snapshot(nfs_root, 3)
    assert Path(nfs_root, "runs/deferred/.merged").exists()


def _checkpoint_and_crash(root):
    with _local_sqlite_only():
        cache = _cache(root, "crashed")
        cache._store("checkpoint", "generate_until", "nfs_test", 0, 0, {}, "saved")
        cache._checkpoint_to_run_dir()
        os._exit(23)


def test_nfs_checkpoint_survives_process_crash_and_scratch_change(nfs_root, monkeypatch):
    process = multiprocessing.get_context("spawn").Process(target=_checkpoint_and_crash, args=(nfs_root,))
    process.start()
    try:
        process.join(timeout=90)
        assert process.exitcode == 23
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
    assert Path(nfs_root, "runs/crashed/rank_0.db").exists()
    # A new scratch directory simulates rescheduling onto another node.
    with tempfile.TemporaryDirectory(prefix="cache-restarted-") as scratch:
        monkeypatch.setenv("LMMS_LOCAL_CACHE", scratch)
        resumed = _cache(nfs_root, "crashed")
        try:
            assert resumed._lookup("checkpoint") == "saved"
            resumed.finalize(success=True)
        finally:
            resumed.close()
        _assert_snapshot(nfs_root, 1)
