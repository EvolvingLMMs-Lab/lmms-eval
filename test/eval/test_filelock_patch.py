import filelock
from datasets.utils import _filelock as datasets_filelock

from lmms_eval.evaluator import _enable_reentrant_filelocks


def test_enable_reentrant_filelocks_makes_datasets_locks_singleton(tmp_path):
    _enable_reentrant_filelocks()
    _enable_reentrant_filelocks()

    lock_path = tmp_path / "datasets.lock"
    first = datasets_filelock.FileLock(str(lock_path))
    second = datasets_filelock.FileLock(str(lock_path))

    assert first is second

    with first:
        with second:
            assert first.is_locked


def test_cross_class_singleton_no_deadlock(tmp_path):
    """filelock.FileLock and datasets.FileLock on the same path must share
    one instance so the global _registry deadlock check never fires."""
    _enable_reentrant_filelocks()

    lock_path = tmp_path / "cross_class.lock"
    fl_instance = filelock.FileLock(str(lock_path))
    ds_instance = datasets_filelock.FileLock(str(lock_path))

    # Cross-class cache must return the exact same object
    assert fl_instance is ds_instance

    with fl_instance:
        with ds_instance:
            assert fl_instance.is_locked
