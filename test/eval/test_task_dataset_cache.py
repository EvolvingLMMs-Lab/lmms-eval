from lmms_eval.api import task as task_module


def _clear_cache_dir_resolver_cache():
    task_module._resolve_hf_datasets_cache_dir.cache_clear()


def test_resolve_hf_datasets_cache_dir_prefers_explicit_override(tmp_path, monkeypatch):
    explicit_cache_dir = tmp_path / "explicit"
    monkeypatch.setenv("LMMS_EVAL_DATASETS_CACHE", str(explicit_cache_dir))
    monkeypatch.setenv("HF_DATASETS_CACHE", "/mnt/vast/cache/huggingface/datasets")
    monkeypatch.setattr(task_module, "detect_fs_type", lambda path: task_module.FsType.REMOTE)
    monkeypatch.setattr(task_module, "find_local_scratch", lambda: str(tmp_path / "scratch"))

    _clear_cache_dir_resolver_cache()
    resolved = task_module._resolve_hf_datasets_cache_dir()
    _clear_cache_dir_resolver_cache()

    assert resolved == str(explicit_cache_dir)
    assert explicit_cache_dir.is_dir()


def test_resolve_hf_datasets_cache_dir_redirects_remote_cache_to_local_scratch(tmp_path, monkeypatch):
    scratch_root = tmp_path / "scratch"
    monkeypatch.delenv("LMMS_EVAL_DATASETS_CACHE", raising=False)
    monkeypatch.setenv("HF_DATASETS_CACHE", "/mnt/vast/cache/huggingface/datasets")
    monkeypatch.setenv("USER", "brian")
    monkeypatch.setattr(task_module, "detect_fs_type", lambda path: task_module.FsType.REMOTE)
    monkeypatch.setattr(task_module, "find_local_scratch", lambda: str(scratch_root))

    _clear_cache_dir_resolver_cache()
    resolved = task_module._resolve_hf_datasets_cache_dir()
    _clear_cache_dir_resolver_cache()

    expected = scratch_root / "lmms_eval_hf_datasets" / "brian"
    assert resolved == str(expected)
    assert expected.is_dir()


def test_resolve_hf_datasets_cache_dir_keeps_remote_cache_when_no_local_scratch(tmp_path, monkeypatch):
    remote_cache_dir = tmp_path / "remote-cache"
    monkeypatch.delenv("LMMS_EVAL_DATASETS_CACHE", raising=False)
    monkeypatch.setenv("HF_DATASETS_CACHE", str(remote_cache_dir))
    monkeypatch.setattr(task_module, "detect_fs_type", lambda path: task_module.FsType.REMOTE)
    monkeypatch.setattr(task_module, "find_local_scratch", lambda: None)

    _clear_cache_dir_resolver_cache()
    resolved = task_module._resolve_hf_datasets_cache_dir()
    _clear_cache_dir_resolver_cache()

    assert resolved == str(remote_cache_dir)
    assert remote_cache_dir.is_dir()
