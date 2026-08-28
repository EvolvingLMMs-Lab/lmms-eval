import threading
import time
from types import SimpleNamespace

import pytest

from lmms_eval.api.instance import Instance, TokenCounts
from lmms_eval.evaluator import _process_document_results, _thread_map_ordered


def test_thread_map_runs_concurrently_and_returns_input_order():
    barrier = threading.Barrier(4, timeout=5)
    completion_order = []
    completion_lock = threading.Lock()

    def work(value):
        barrier.wait()
        time.sleep((3 - value) * 0.01)
        with completion_lock:
            completion_order.append(value)
        return value * 10

    results = _thread_map_ordered(work, range(4), max_workers=4)

    assert completion_order != [0, 1, 2, 3]
    assert results == [0, 10, 20, 30]


def test_thread_map_one_worker_is_serial():
    completion_order = []

    def work(value):
        completion_order.append(value)
        return value

    assert _thread_map_ordered(work, range(4), max_workers=1) == [0, 1, 2, 3]
    assert completion_order == [0, 1, 2, 3]


def test_thread_map_rejects_non_positive_worker_count():
    with pytest.raises(ValueError, match="at least 1"):
        _thread_map_ordered(lambda value: value, [], max_workers=0)


def test_process_document_results_preserves_serial_scoring_and_logging_contract():
    doc = {"question": "What is the answer?", "target": "Paris", "image": b"binary-image"}
    request = Instance(
        request_type="generate_until",
        arguments=("prompt", {"until": ["stop"]}, None, 5, "test_task", "test"),
        idx=0,
        metadata={"task": "test_task", "doc_id": 5, "repeats": 1},
    )
    request.doc = doc
    request.resps = ["<think>private chain</think> Paris"]
    request.filtered_resps["default"] = "<think>private chain</think> Paris"
    request.token_counts = [TokenCounts(input_tokens=11, output_tokens=7)]

    seen_results = []

    class FakeTask:
        config = SimpleNamespace(repeats=1)

        def process_results(self, current_doc, results):
            assert current_doc is doc
            seen_results.append(results)
            return {"accuracy": float(results == ["Paris"])}

        def doc_to_target(self, current_doc):
            return current_doc["target"]

    processed = _process_document_results(
        (5, doc, [request]),
        task=FakeTask(),
        filter_key="default",
        reasoning_tags=[["<think>", "</think>"]],
        log_samples=True,
    )

    assert seen_results == [["Paris"]]
    assert processed.metrics == {"accuracy": 1.0}
    assert processed.per_sample_scores == {}
    assert processed.logged_sample["doc_id"] == 5
    assert processed.logged_sample["doc"] == {"question": "What is the answer?", "target": "Paris"}
    assert processed.logged_sample["target"] == "Paris"
    assert processed.logged_sample["resps"] == ["<think>private chain</think> Paris"]
    assert processed.logged_sample["filtered_resps"] == ["Paris"]
    assert processed.logged_sample["token_counts"] == [{"input_tokens": 11, "output_tokens": 7}]
    assert processed.logged_sample["accuracy"] == 1.0


def test_process_document_results_computes_repeat_scores():
    doc = {"target": "A"}
    requests = []
    for index, response in enumerate(("A", "B")):
        request = Instance(
            request_type="generate_until",
            arguments=("prompt", {}, None, 3, "test_task", "test"),
            idx=index,
            metadata={"task": "test_task", "doc_id": 3, "repeats": 2},
        )
        request.doc = doc
        request.filtered_resps["default"] = response
        requests.append(request)

    class FakeTask:
        config = SimpleNamespace(repeats=2)

        def process_results(self, current_doc, results):
            return {"accuracy": float(results[0] == current_doc["target"])}

    processed = _process_document_results(
        (3, doc, requests),
        task=FakeTask(),
        filter_key="default",
        reasoning_tags=None,
        log_samples=False,
    )

    assert processed.metrics == {"accuracy": 1.0}
    assert processed.per_sample_scores == {"accuracy": [1.0, 0.0]}
    assert processed.logged_sample is None
