"""Production request-construction contracts without dataset initialization."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lmms_eval.api.task import ConfigurableMessagesTask, ConfigurableTask, TaskConfig
from lmms_eval.caching import cache as request_cache


def _doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return "prompt"


def _task(task_class, output_type):
    """Allocate a task shell so construct_requests never loads a dataset."""
    task = object.__new__(task_class)
    task.OUTPUT_TYPE = output_type
    task._config = SimpleNamespace(
        task="request-contract",
        generation_kwargs={"temperature": 0, "until": ["<stop>"]},
        doc_to_text=_doc_to_text,
    )
    task.lmms_eval_specific_kwargs = {"backend": "contract"}
    return task


def _request(task):
    return task.construct_requests(
        17,
        "context",
        metadata={"task": "request-contract", "doc_id": 17, "repeats": 1, "split": "validation"},
    )


def test_configurable_task_generate_until_constructs_simple_layout():
    task = _task(ConfigurableTask, "generate_until")

    request = _request(task)

    assert request.request_type == "generate_until"
    assert len(request.args) == 6
    assert request.args[0] == "context"
    assert isinstance(request.args[1], dict)
    assert request.args[1] == {"temperature": 0, "until": ["<stop>"]}
    assert request.args[1] is not task.config.generation_kwargs
    assert callable(request.args[2])
    assert request.args[3] == 17
    assert request.args[4] == "request-contract"
    assert request.args[5] == "validation"


def test_configurable_task_loglikelihood_constructs_simple_layout():
    task = _task(ConfigurableTask, "loglikelihood")

    request = _request(task)

    assert request.request_type == "loglikelihood"
    assert len(request.args) == 6
    assert request.args[0] == "context"
    assert callable(request.args[1])
    assert callable(request.args[2])
    assert request.args[3] == 17
    assert request.args[4] == "request-contract"
    assert request.args[5] == "validation"


@pytest.mark.parametrize("output_type", ["generate_until_multi_round", "generate_until_agentic"])
def test_configurable_task_constructs_multi_step_layout(output_type):
    task = _task(ConfigurableTask, output_type)

    request = _request(task)

    assert request.request_type == output_type
    assert len(request.args) == 7
    assert request.args[0] == "context"
    assert isinstance(request.args[1], dict)
    assert request.args[1] == {"temperature": 0, "until": ["<stop>"]}
    assert callable(request.args[2])
    assert callable(request.args[3])
    assert request.args[3].func is _doc_to_text
    assert request.args[3].keywords == {"lmms_eval_specific_kwargs": {"backend": "contract"}}
    assert request.args[4] == 17
    assert request.args[5] == "request-contract"
    assert request.args[6] == "validation"


@pytest.mark.parametrize("output_type", ["generate_until", "generate_until_multi_round"])
def test_messages_task_constructs_chat_generate_layout(output_type):
    task = _task(ConfigurableMessagesTask, output_type)

    request = _request(task)

    assert request.request_type == output_type
    assert len(request.args) == 6
    assert request.args[0] == "context"
    assert callable(request.args[1])
    assert isinstance(request.args[2], dict)
    assert request.args[2] == {"temperature": 0, "until": ["<stop>"]}
    assert request.args[2] is not task.config.generation_kwargs
    assert request.args[3] == 17
    assert request.args[4] == "request-contract"
    assert request.args[5] == "validation"


def test_messages_task_agentic_constructs_chat_multi_step_layout():
    task = _task(ConfigurableMessagesTask, "generate_until_agentic")

    request = _request(task)

    assert request.request_type == "generate_until_agentic"
    assert len(request.args) == 7
    assert request.args[0] == "context"
    assert isinstance(request.args[1], dict)
    assert request.args[1] == {"temperature": 0, "until": ["<stop>"]}
    assert callable(request.args[2])
    assert callable(request.args[3])
    assert request.args[3].func is _doc_to_text
    assert request.args[3].keywords == {"lmms_eval_specific_kwargs": {"backend": "contract"}}
    assert request.args[4] == 17
    assert request.args[5] == "request-contract"
    assert request.args[6] == "validation"


def _cached_task(task_class=ConfigurableTask, output_type="generate_until"):
    task = object.__new__(task_class)
    task.OUTPUT_TYPE = output_type
    task._config = TaskConfig(
        task="cache-contract",
        output_type=output_type,
        test_split="test",
        num_fewshot=0,
        doc_to_text=_doc_to_text,
        doc_to_visual=lambda doc: [doc["media"]],
        doc_to_target=lambda doc: doc["answer"],
        doc_to_messages=lambda doc: [{"role": "user", "content": [{"type": "text", "text": doc["question"]}]}],
        generation_kwargs={"temperature": 0},
    )
    task.lmms_eval_specific_kwargs = None
    task.model_specific_target_kwargs = None
    task.dataset = {"test": [{"question": f"question{i}", "answer": "yes", "media": f"media{i}"} for i in range(6)]}
    task.dataset_no_image = task.dataset
    task.fewshot_context = Mock(side_effect=lambda doc, *args: doc["question"])
    return task


@pytest.mark.parametrize(
    "task_class,output_type",
    [
        (ConfigurableTask, "generate_until"),
        (ConfigurableTask, "loglikelihood"),
        (ConfigurableTask, "generate_until_multi_round"),
        (ConfigurableTask, "generate_until_agentic"),
        (ConfigurableMessagesTask, "generate_until"),
        (ConfigurableMessagesTask, "generate_until_multi_round"),
        (ConfigurableMessagesTask, "generate_until_agentic"),
    ],
)
def test_request_cache_cold_and_warm_preserve_live_callbacks(tmp_path, monkeypatch, task_class, output_type):
    monkeypatch.setattr(request_cache, "PATH", str(tmp_path))
    cold = _cached_task(task_class, output_type)
    cold.build_all_requests(cache_requests=True, limit=2)
    expected = _cached_task(task_class, output_type)
    expected.build_all_requests(cache_requests=False, limit=2)
    warm = _cached_task(task_class, output_type)
    warm.build_all_requests(cache_requests=True, limit=2)
    warm.fewshot_context.assert_not_called()

    for task in (cold, warm):
        assert [request.doc_id for request in task.instances] == [0, 1]
        for actual, reference in zip(task.instances, expected.instances):
            doc = task.dataset["test"][actual.doc_id]
            assert len(actual.args) == len(reference.args)
            for arg, ref in zip(actual.args, reference.args):
                if callable(ref):
                    assert callable(arg)
                    assert arg(doc) == ref(doc)
                    if hasattr(arg, "__self__"):
                        assert arg.__self__ is task
                else:
                    assert arg == ref


@pytest.mark.parametrize("enabled,refresh", [(False, False), (False, True), (True, True)])
def test_disabled_or_refresh_request_cache_never_reads(monkeypatch, tmp_path, enabled, refresh):
    monkeypatch.setattr(request_cache, "PATH", str(tmp_path))
    load = Mock(side_effect=AssertionError("must not deserialize a cache"))
    monkeypatch.setattr("lmms_eval.api.task.load_from_cache", load)
    task = _cached_task()
    task.build_all_requests(cache_requests=enabled, rewrite_requests_cache=refresh, limit=2)
    load.assert_not_called()
    assert [request.doc_id for request in task.instances] == [0, 1]
    assert bool(list(tmp_path.iterdir())) is enabled


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_request_cache_preserves_global_limit_and_offset(tmp_path, monkeypatch, rank):
    monkeypatch.setattr(request_cache, "PATH", str(tmp_path))
    for enabled in (False, True, True):
        task = _cached_task()
        task.build_all_requests(cache_requests=enabled, limit=4, offset=1, rank=rank, world_size=3)
        assert [request.doc_id for request in task.instances] == list(range(1 + rank, 5, 3))


def test_request_cache_separates_simple_and_chat_tasks(tmp_path, monkeypatch):
    monkeypatch.setattr(request_cache, "PATH", str(tmp_path))
    simple = _cached_task()
    simple.build_all_requests(cache_requests=True, limit=1)
    chat = _cached_task(ConfigurableMessagesTask)
    chat.build_all_requests(cache_requests=True, limit=1)
    assert chat.fewshot_context.called
    assert len(list(tmp_path.iterdir())) == 2


def test_request_cache_bounds_filename_with_system_instruction(tmp_path, monkeypatch):
    monkeypatch.setattr(request_cache, "PATH", str(tmp_path))
    for _ in range(2):
        task = _cached_task(ConfigurableMessagesTask)
        task.build_all_requests(cache_requests=True, limit=1, system_instruction="Be precise", tokenizer_name="organization/" + "model" * 60)
        assert len(task.instances) == 1
    task.fewshot_context.assert_not_called()
    assert len(next(tmp_path.iterdir()).name.encode()) <= 255


def test_request_cache_save_preserves_contexts_and_previous_file(tmp_path, monkeypatch):
    monkeypatch.setattr(request_cache, "PATH", str(tmp_path / "nested" / "cache"))
    contexts = [{"doc_id": 0, "context": "original"}]
    request_cache.save_to_cache("test", contexts)
    assert request_cache.load_from_cache("test") == contexts
    path = next((tmp_path / "nested" / "cache").iterdir())
    before = path.read_bytes()
    monkeypatch.setattr(request_cache.os, "replace", Mock(side_effect=OSError("publication failed")))
    with pytest.raises(OSError, match="publication failed"):
        request_cache.save_to_cache("test", [{"doc_id": 0, "context": "replacement"}])
    assert path.read_bytes() == before
    assert list(path.parent.iterdir()) == [path]


def test_request_cache_rebuilds_current_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(request_cache, "PATH", str(tmp_path))
    cold = _cached_task()
    cold.config.metadata = {"sample_frames": 8, "old_key": True, "repeats": 3}
    cold.build_all_requests(cache_requests=True, limit=1)
    for metadata in (cold.config.metadata, {"sample_frames": 32, "repeats": 2}):
        warm = _cached_task()
        warm.config.metadata = metadata
        warm.build_all_requests(cache_requests=True, limit=1)
        reference = _cached_task()
        reference.config.metadata = metadata
        reference.build_all_requests(limit=1)
        assert warm.instances[0].metadata == reference.instances[0].metadata
        assert warm.instances[0].repeats == metadata["repeats"]


def test_request_cache_preserves_context_before_custom_construction(tmp_path, monkeypatch):
    class PrefixTask(ConfigurableTask):
        def construct_requests(self, doc_id, ctx, **kwargs):
            return super().construct_requests(doc_id, f"prefix {ctx}", **kwargs)

    monkeypatch.setattr(request_cache, "PATH", str(tmp_path))
    for _ in range(2):
        task = _cached_task(PrefixTask)
        task.build_all_requests(cache_requests=True, limit=1)
        assert task.instances[0].args[0] == "prefix question0"
    task.fewshot_context.assert_not_called()
