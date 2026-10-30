"""Publication, reader isolation, and recovery regressions for issue #1614."""

import multiprocessing
import os
import sqlite3
from unittest.mock import patch

import pytest

from lmms_eval.caching import response_cache as module
from lmms_eval.caching.fs_detect import FsType, find_local_scratch
from lmms_eval.caching.response_cache import ResponseCache


def _shard(path, key):
    cache = ResponseCache(str(path), str(path) + ".jsonl")
    cache._store(key, "generate_until", "test", 0, 0, {}, key)
    cache.close()


def _publish_run(root, run):
    path = os.path.join(root, "runs", run, "rank_0.db")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _shard(path, run)
    cache = ResponseCache(path, path + ".jsonl")
    cache._cache_root = root
    cache._run_dir = os.path.dirname(path)
    cache.finalize(success=True)


def _count(path):
    db = sqlite3.connect(path)
    try:
        assert db.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert db.execute("PRAGMA journal_mode").fetchone() == ("delete",)
        return db.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
    finally:
        db.close()


def test_reader_pins_private_snapshot_across_publications(tmp_path):
    first, second, root = [tmp_path / name for name in ("first.db", "second.db", "cache.db")]
    _shard(first, "first")
    _shard(second, "second")
    ResponseCache.merge_shards([str(first)], str(root))
    reader = ResponseCache(str(tmp_path / "reader.db"), str(tmp_path / "reader.jsonl"), shared_db_path=str(root))
    private = reader._reader_scratch.name
    try:
        ResponseCache.merge_shards([str(second)], str(root))
        assert reader._lookup("first") == "first"
        assert reader._lookup("second") is None
        assert _count(root) == 2
        assert not os.path.exists(str(root) + "-wal")
        assert not os.path.exists(str(root) + "-shm")
    finally:
        reader.close()
    assert not os.path.exists(private)
    fresh = ResponseCache(str(tmp_path / "fresh.db"), str(tmp_path / "fresh.jsonl"), shared_db_path=str(root))
    try:
        assert fresh._lookup("second") == "second"
    finally:
        fresh.close()


def test_parallel_runs_keep_all_entries(tmp_path):
    context = multiprocessing.get_context("spawn")
    workers = [context.Process(target=_publish_run, args=(str(tmp_path), str(index))) for index in range(4)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(30)
        if worker.is_alive():
            worker.terminate()
            worker.join()
            pytest.fail("cache publisher did not finish")
        assert worker.exitcode == 0
    assert _count(tmp_path / "cache.db") == 4


def test_failed_publication_preserves_old_snapshot_and_ready_run(tmp_path):
    _publish_run(str(tmp_path), "first")
    original_replace = os.replace

    def fail_snapshot(source, destination):
        if str(destination) == str(tmp_path / "cache.db"):
            raise OSError("injected publication failure")
        return original_replace(source, destination)

    with patch.object(module.os, "replace", side_effect=fail_snapshot):
        with pytest.raises(OSError, match="injected"):
            _publish_run(str(tmp_path), "second")
    assert _count(tmp_path / "cache.db") == 1
    assert (tmp_path / "runs/second/.ready").exists()
    assert not (tmp_path / "runs/second/.merged").exists()
    assert not list(tmp_path.glob(".cache-*.tmp"))
    _publish_run(str(tmp_path), "third")
    assert _count(tmp_path / "cache.db") == 3
    assert (tmp_path / "runs/second/.merged").exists()


def test_remote_merge_never_opens_remote_sqlite(tmp_path):
    remote, local = tmp_path / "remote", tmp_path / "local"
    remote.mkdir()
    local.mkdir()
    _shard(remote / "rank.db", "answer")
    connect = sqlite3.connect

    def local_connect(path, *args, **kwargs):
        assert not str(path).startswith(str(remote))
        return connect(path, *args, **kwargs)

    with (
        patch.object(module, "detect_fs_type", side_effect=lambda path: FsType.REMOTE if str(path).startswith(str(remote)) else FsType.LOCAL),
        patch.object(module, "find_local_scratch", return_value=str(local)),
        patch.object(module.sqlite3, "connect", side_effect=local_connect),
    ):
        ResponseCache.merge_shards([str(remote / "rank.db")], str(remote / "cache.db"))
    assert _count(remote / "cache.db") == 1


def test_remote_without_scratch_fails_before_sqlite(tmp_path):
    with (
        patch.object(module, "detect_fs_type", return_value=FsType.REMOTE),
        patch.object(module, "find_local_scratch", return_value=None),
        patch.object(module.sqlite3, "connect") as connect,
    ):
        with pytest.raises(RuntimeError, match="LMMS_LOCAL_CACHE"):
            ResponseCache.create(str(tmp_path))
        with pytest.raises(RuntimeError, match="LMMS_LOCAL_CACHE"):
            ResponseCache.merge_shards([], str(tmp_path / "cache.db"))
        connect.assert_not_called()


def test_scratch_rejects_remote_and_unknown_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("LMMS_LOCAL_CACHE", str(tmp_path))
    for kind in (FsType.REMOTE, FsType.UNKNOWN):
        with patch("lmms_eval.caching.fs_detect.detect_fs_type", return_value=kind):
            assert find_local_scratch() is None


def test_checkpoint_is_closed_atomic_and_recoverable(tmp_path, monkeypatch):
    remote, local = tmp_path / "remote", tmp_path / "local"
    local.mkdir()
    monkeypatch.setenv("LMMS_CACHE_RUN_ID", "restart")
    with (
        patch.object(module, "detect_fs_type", side_effect=lambda path: FsType.REMOTE if str(path).startswith(str(remote)) else FsType.LOCAL),
        patch.object(module, "find_local_scratch", return_value=str(local)),
    ):
        cache = ResponseCache.create(str(remote))
        cache._store("saved", "generate_until", "test", 0, 0, {}, "saved")
        cache._checkpoint_to_run_dir()
        checkpoint = cache._remote_rank_db
        cache.close()
        assert _count(checkpoint) == 1
        # Simulate restart on a different node without the old scratch files.
        other = tmp_path / "other-local"
        other.mkdir()
        with patch.object(module, "find_local_scratch", return_value=str(other)):
            restarted = ResponseCache.create(str(remote))
            try:
                assert restarted._lookup("saved") == "saved"
                restarted.finalize(success=True)
            finally:
                restarted.close()
    assert _count(remote / "cache.db") == 1


def test_legacy_live_wal_is_rejected(tmp_path):
    source = tmp_path / "active.db"
    cache = ResponseCache(str(source), str(source) + ".jsonl")
    try:
        cache._store("saved", "generate_until", "test", 0, 0, {}, "saved")
        with pytest.raises(RuntimeError, match="checkpoint it locally"):
            ResponseCache.merge_shards([str(source)], str(tmp_path / "cache.db"))
    finally:
        cache.close()


def test_lock_timeout_keeps_run_ready_for_later_merge(tmp_path):
    with patch.object(module, "_merge_lock", side_effect=TimeoutError("busy")):
        _publish_run(str(tmp_path), "deferred")
    assert (tmp_path / "runs/deferred/.ready").exists()
    assert not (tmp_path / "runs/deferred/.merged").exists()
    _publish_run(str(tmp_path), "next")
    assert _count(tmp_path / "cache.db") == 2


def test_failed_checkpoint_keeps_previous_checkpoint(tmp_path):
    cache = ResponseCache(str(tmp_path / "writer.db"), str(tmp_path / "writer.jsonl"))
    cache._use_scratch = True
    cache._remote_rank_db = str(tmp_path / "checkpoint.db")
    try:
        cache._store("first", "generate_until", "test", 0, 0, {}, "first")
        cache._checkpoint_to_run_dir()
        cache._store("second", "generate_until", "test", 1, 0, {}, "second")
        with patch.object(module.os, "replace", side_effect=OSError("disk unavailable")):
            with pytest.raises(RuntimeError, match="checkpoint"):
                cache.finalize(success=True)
        assert _count(tmp_path / "checkpoint.db") == 1
        assert not list(tmp_path.glob(".cache-*.tmp"))
    finally:
        cache.close()


def _crash_with_wal(path):
    cache = ResponseCache(path, path + ".jsonl")
    cache._store("wal", "generate_until", "test", 1, 0, {}, "wal")
    os._exit(0)


def test_restart_recovers_local_wal_before_merging_checkpoint(tmp_path, monkeypatch):
    remote, local = tmp_path / "remote", tmp_path / "local"
    local.mkdir()
    monkeypatch.setenv("LMMS_CACHE_RUN_ID", "crashed")
    with (
        patch.object(module, "detect_fs_type", side_effect=lambda path: FsType.REMOTE if str(path).startswith(str(remote)) else FsType.LOCAL),
        patch.object(module, "find_local_scratch", return_value=str(local)),
    ):
        cache = ResponseCache.create(str(remote))
        cache._store("checkpoint", "generate_until", "test", 0, 0, {}, "checkpoint")
        cache._checkpoint_to_run_dir()
        path = cache.db_path
        cache.close()
        process = multiprocessing.get_context("spawn").Process(target=_crash_with_wal, args=(path,))
        process.start()
        process.join(30)
        assert process.exitcode == 0
        assert os.path.exists(path + "-wal")
        resumed = ResponseCache.create(str(remote))
        try:
            assert resumed._lookup("checkpoint") == "checkpoint"
            assert resumed._lookup("wal") == "wal"
            resumed.finalize(success=True)
        finally:
            resumed.close()
    assert _count(remote / "cache.db") == 2
