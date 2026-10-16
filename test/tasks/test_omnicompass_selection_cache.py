"""Subset changes must not reuse another question's answer, even for equal prompts."""

from types import SimpleNamespace

import pytest

from lmms_eval.caching.response_cache import ResponseCache
from lmms_eval.tasks import TaskManager
from lmms_eval.tasks.omnicompass import task as task_module
from lmms_eval.tasks.omnicompass import utils
from lmms_eval.tasks.omnicompass._core.bridge import Protocol


@pytest.mark.parametrize("merge", [False, True])
def test_same_directory_changed_subset_and_resume(tmp_path, monkeypatch, merge):
    # Keep model/version/protocol fixed so only the selection can prevent a hit.
    monkeypatch.setattr("lmms_eval.utils.get_lmms_eval_cache_version", lambda: "fixed")
    adapter = object.__new__(Protocol)
    adapter.protocol_id = "fixed-protocol"
    adapter.store = SimpleNamespace(local=True)
    adapter.videos = {vid: {"duration_seconds": 30} for vid in ("oc_071", "oc_073")}
    adapter.questions = [
        {
            "question_id": f"{vid}_vl_tg_01",
            "video_id": vid,
            "track": "VL",
            "task": "TG",
            "question": {"text": f"When does {vid} act?", "text_image": "When does the action in the reference image occur?"},
            "ground_truth": {"start_time": 3, "end_time": end},
        }
        for vid, end in (("oc_071", 5), ("oc_073", 7))
    ]
    monkeypatch.setattr(task_module, "protocol", lambda: adapter)
    monkeypatch.setattr(utils, "protocol", lambda: adapter)
    monkeypatch.delenv("OMNICOMPASS_REPORT_DIR", raising=False)
    manager = TaskManager(model_name="openai")
    fingerprints = {}
    prompts = {}

    def run(selection, expected_calls):
        monkeypatch.setenv("OMNICOMPASS_VIDEO_IDS", selection)
        tasks = manager.load_task_or_group(["omnicompass_vl_tg"])
        task = tasks["omnicompass_vl_tg"]
        task.build_all_requests(limit=None, rank=0, world_size=1)
        docs = task.test_docs()
        expected = [f"{doc['question_id']}/{doc['setting']}" for doc in docs]
        prompts[selection] = [instance.args[0] for instance in task.instances]
        fingerprints[selection] = task.dump_config()["metadata"]["selection_sha256"]
        calls = []

        class Model:
            def generate_until(self, batch):
                answers = [f"{docs[req.doc_id]['question_id']}/{docs[req.doc_id]['setting']}" for req in batch]
                calls.extend(answers)
                return answers

        cache = ResponseCache.create(str(tmp_path), model="openai", model_args="model_version=fake", task_dict=tasks)
        try:
            assert cache.execute(Model(), "generate_until", task.instances) == expected
            assert len(calls) == expected_calls
            assert cache.get_stats()["misses"] == expected_calls
            assert cache.get_stats()["hits"] == len(expected) - expected_calls
        finally:
            cache.finalize(success=merge)

    run("oc_071", 2)
    run("oc_073", 2)
    assert prompts["oc_071"][1] == prompts["oc_073"][1]
    assert fingerprints["oc_071"] != fingerprints["oc_073"]
    run("oc_073", 0)
    run("oc_071,oc_073", 4)
    # Selector spelling/order does not change the actual dataset order.
    run(" oc_073, oc_071 ", 0)
    assert fingerprints["oc_071,oc_073"] == fingerprints[" oc_073, oc_071 "]
    run("oc_071", 0)
    # A changed dataset order must invalidate positional doc IDs as well.
    adapter.questions.reverse()
    run("oc_071,oc_073", 4)
