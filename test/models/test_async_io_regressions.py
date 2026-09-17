"""Exercise async I/O without importing optional model or TUI dependencies."""

import ast
import asyncio
import os
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_functions(relative_path, names, namespace, class_name=None):
    # Like test_decord_lazy_imports, isolate production methods from heavyweight imports.
    path = ROOT / relative_path
    tree = ast.parse(path.read_text())
    body = tree.body
    if class_name is not None:
        body = next(node for node in body if isinstance(node, ast.ClassDef) and node.name == class_name).body
    functions = [node for node in body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    assert {node.name for node in functions} == set(names)
    for function in functions:
        function.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *functions], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


@pytest.fixture
def omni():
    namespace = _load_functions(
        "lmms_eval/models/chat/vllm_omni_api.py",
        ("_cached_video_path", "_write_video", "_post_one"),
        {"asyncio": asyncio, "os": os, "time": time, "eval_logger": Mock()},
        class_name="VLLMOmniAPI",
    )
    model = SimpleNamespace(
        overwrite=False,
        base_urls=["http://example.invalid"],
        max_retries=2,
        retry_backoff_s=0,
        _image_file_tuple=lambda image: None,
        _image_reference_json=lambda image: None,
        _endpoint_url=lambda base_url: base_url,
        _headers=lambda: {},
        _pack_result=lambda path, metadata: {"path": path, "metadata": metadata},
        _empty_result=lambda error: {"error": error},
        _write_video=namespace["_write_video"],
    )
    model._cached_video_path = lambda path: namespace["_cached_video_path"](model, path)
    model._post_one = lambda client, prep, idx: namespace["_post_one"](model, client, prep, idx)
    return model


@pytest.mark.parametrize(("content", "overwrite", "cached"), [(b"video", False, True), (b"", False, False), (b"video", True, False)])
def test_omni_cache_policy(omni, tmp_path, content, overwrite, cached):
    path = tmp_path / "cached.mp4"
    assert omni._cached_video_path(str(path)) is None
    path.write_bytes(content)
    omni.overwrite = overwrite
    assert omni._cached_video_path(str(path)) == (str(path.resolve()) if cached else None)


@pytest.mark.parametrize("cached", [False, True])
def test_omni_file_io_runs_off_event_loop(omni, tmp_path, cached):
    path = tmp_path / "nested" / "video.mp4"
    if cached:
        path.parent.mkdir()
        path.write_bytes(b"cached")
    loop_thread = threading.get_ident()
    io_threads = []

    def track(function):
        def call(*args):
            io_threads.append(threading.get_ident())
            return function(*args)

        return call

    omni._cached_video_path = track(omni._cached_video_path)
    omni._write_video = track(omni._write_video)
    client = SimpleNamespace(post=AsyncMock(return_value=SimpleNamespace(status_code=200, content=b"generated", headers={})))
    result, idx = asyncio.run(omni._post_one(client, {"output_path": str(path), "prompt": "test", "params": {}}, 0))

    assert idx == 0
    assert result["path"] == str(path.resolve())
    assert result["metadata"]["cached"] is cached
    assert path.read_bytes() == (b"cached" if cached else b"generated")
    assert len(io_threads) == (1 if cached else 2)
    assert all(thread != loop_thread for thread in io_threads)
    assert client.post.await_count == (0 if cached else 1)


def test_omni_write_failure_keeps_retry_behavior(omni, tmp_path):
    omni._write_video = Mock(side_effect=OSError("disk full"))
    client = SimpleNamespace(post=AsyncMock(return_value=SimpleNamespace(status_code=200, content=b"video", headers={})))
    result, idx = asyncio.run(omni._post_one(client, {"output_path": str(tmp_path / "video.mp4"), "prompt": "test", "params": {}}, 0))

    assert result == {"error": "disk full"}
    assert idx == 0
    assert client.post.await_count == 2
    assert omni._write_video.call_count == 2


def test_srt_retry_awaits_sleep():
    sleeper = AsyncMock()
    namespace = _load_functions(
        "lmms_eval/models/simple/srt_api.py",
        ("generate",),
        {"asyncio": SimpleNamespace(sleep=sleeper), "eval_logger": Mock(), "NUM_SECONDS_TO_SLEEP": 5},
        class_name="SRT_API",
    )
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=" answer "))])
    create = AsyncMock(side_effect=[RuntimeError("retry"), response])
    model = SimpleNamespace(
        task_dict={"task": {"test": [{}]}},
        flatten=lambda groups: [item for group in groups for item in group],
        modality="image",
        add_time_instruction=False,
        model_version="test",
        timeout=1,
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    )
    request = SimpleNamespace(args=("question", {}, lambda doc: [], 0, "task", "test"))

    assert asyncio.run(namespace["generate"](model, request)) == "answer"
    assert create.await_count == 2
    sleeper.assert_awaited_once_with(5)


@pytest.mark.parametrize(("task_id", "status"), [("example", None), ("missing", 404)])
def test_tui_yaml_lookup_runs_off_event_loop(tmp_path, task_id, status):
    class HTTPException(Exception):
        def __init__(self, status_code, detail):
            super().__init__(detail)
            self.status_code = status_code

    tasks = tmp_path / "lmms_eval" / "tasks"
    tasks.mkdir(parents=True)
    content = "task: example\ndataset_path: test\n"
    (tasks / "example.yaml").write_text(content)
    namespace = _load_functions(
        "lmms_eval/tui/server.py",
        ("get_task_yaml", "_get_task_yaml"),
        {"asyncio": asyncio, "Path": Path, "re": re, "HTTPException": HTTPException, "__file__": str(tmp_path / "lmms_eval" / "tui" / "server.py")},
    )
    lookup = namespace["_get_task_yaml"]
    loop_thread = threading.get_ident()
    io_threads = []

    def tracked_lookup(value):
        io_threads.append(threading.get_ident())
        return lookup(value)

    namespace["_get_task_yaml"] = tracked_lookup
    if status is None:
        result = asyncio.run(namespace["get_task_yaml"](task_id))
        assert result == {"task_id": "example", "yaml": content, "path": "lmms_eval/tasks/example.yaml"}
    else:
        with pytest.raises(HTTPException) as exc:
            asyncio.run(namespace["get_task_yaml"](task_id))
        assert exc.value.status_code == status
    assert len(io_threads) == 1
    assert io_threads[0] != loop_thread
