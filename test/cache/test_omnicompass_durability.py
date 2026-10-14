"""OmniCompass interruption, revision and checkpoint-failure regression tests."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from lmms_eval.api.instance import Instance
from lmms_eval.caching.fs_detect import FsType
from lmms_eval.caching.response_cache import ResponseCache


class FixtureTask:
    def dump_config(self):
        return {"task": "omnicompass_vl_qa", "fixture": True}


def requests():
    return [Instance(request_type="generate_until", arguments=(str(i), {"temperature": 0}, None, i, "omnicompass_vl_qa", "test"), idx=0, metadata={"task": "omnicompass_vl_qa", "doc_id": i, "repeats": 1}) for i in range(3)]


def cache(root, **kwargs):
    return ResponseCache.create(str(root), model="fixture", model_args={"pretrained": "fixture/qwen"}, task_dict={"omnicompass_vl_qa": FixtureTask()}, **kwargs)


class FixtureModel:
    def __init__(self, root):
        self.root = root

    def generate_until(self, batch):
        result = []
        for req in batch:
            with (self.root / "calls.jsonl").open("a") as file:
                file.write(json.dumps(req.doc_id) + "\n")
                file.flush()
                os.fsync(file.fileno())
            if req.doc_id == 1 and os.getenv("INTERRUPT_FIXTURE") == "1":
                (self.root / "ready").touch()
                time.sleep(60)
            result.append(f"answer-{req.doc_id}")
        return result


@pytest.mark.parametrize("remote", [False, True])
def test_kill_resume_and_changed_job_id(tmp_path, remote):
    command = [sys.executable, __file__, "--worker", str(tmp_path), str(int(remote))]
    env = {**os.environ, "INTERRUPT_FIXTURE": "1", "SLURM_JOB_ID": "old-job", "OMNICOMPASS_CHECKPOINT_BATCH_SIZE": "1"}
    with (tmp_path / "worker.log").open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 30
            while not (tmp_path / "ready").exists():
                assert process.poll() is None, (tmp_path / "worker.log").read_text()
                assert time.monotonic() < deadline, "Worker did not reach interruption point"
                time.sleep(0.05)
        finally:
            process.kill()
            process.wait(timeout=10)
    env.update(INTERRUPT_FIXTURE="0", SLURM_JOB_ID="new-job")
    resumed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert (tmp_path / "calls.jsonl").read_text().splitlines() == ["0", "1", "1", "2"]
    report = json.loads((tmp_path / "resumed.json").read_text())
    assert report["answers"] == ["answer-0", "answer-1", "answer-2"]
    assert report["stats"]["hits"] == 1
    assert report["stats"]["misses"] == 2
    if remote:
        # Return to node1, whose scratch database still has only the first answer.
        env["INTERRUPT_FIXTURE"] = "1"
        returned = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
        assert returned.returncode == 0, returned.stdout + returned.stderr
        assert (tmp_path / "calls.jsonl").read_text().splitlines() == ["0", "1", "1", "2"]
        assert json.loads((tmp_path / "resumed.json").read_text())["stats"]["hits"] == 3


def test_deployment_change_isolates_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNICOMPASS_MODEL_REVISION", "deployment-v1")
    first = cache(tmp_path)
    identity = first.model_fingerprint
    first.close()
    monkeypatch.setenv("OMNICOMPASS_MODEL_REVISION", "deployment-v2")
    second = cache(tmp_path)
    try:
        assert identity != second.model_fingerprint
    finally:
        second.close()


def test_response_count_mismatch_raises(tmp_path):
    instance = cache(tmp_path)
    try:

        class Model:
            def generate_until(self, batch):
                return []

        with pytest.raises(ValueError, match="number of responses"):
            instance.execute(Model(), "generate_until", requests())
        assert instance.get_stats()["total_cached_entries"] == 0
    finally:
        instance.close()


def test_runtime_change_isolates_cache(tmp_path):
    first = cache(tmp_path)
    identity = first.model_fingerprint
    first.close()
    with patch("lmms_eval.tasks.omnicompass._core.run_identity.runtime_identity", return_value={"packages": {"torch": "changed"}}):
        second = cache(tmp_path)
    try:
        assert identity != second.model_fingerprint
    finally:
        second.close()


def test_concurrent_writer_is_rejected(tmp_path):
    import portalocker

    first = cache(tmp_path)
    try:
        with pytest.raises(portalocker.exceptions.LockException):
            cache(tmp_path)
    finally:
        first.close()


def test_shared_checkpoint_failure_is_fatal(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    with patch("lmms_eval.caching.response_cache.detect_fs_type", return_value=FsType.REMOTE), patch("lmms_eval.caching.response_cache.find_local_scratch", return_value=str(scratch)):
        instance = cache(tmp_path / "cache")
    instance._checkpoint_interval = 1
    try:
        with patch("lmms_eval.caching.response_cache.os.replace", side_effect=OSError("disk full")):
            with pytest.raises(RuntimeError, match="persist"):
                instance.execute(FixtureModel(tmp_path), "generate_until", requests()[:1])
    finally:
        instance.close()


if __name__ == "__main__":
    root, remote = Path(sys.argv[2]), bool(int(sys.argv[3]))
    # Simulate rescheduling onto a different node with no previous local scratch.
    scratch = root / ("node1" if os.getenv("INTERRUPT_FIXTURE") == "1" else "node2")
    scratch.mkdir(exist_ok=True)
    with patch("lmms_eval.caching.response_cache.detect_fs_type", return_value=FsType.REMOTE if remote else FsType.LOCAL), patch("lmms_eval.caching.response_cache.find_local_scratch", return_value=str(scratch)):
        instance = cache(root / "cache")
    try:
        answers = instance.execute(FixtureModel(root), "generate_until", requests())
        (root / "resumed.json").write_text(json.dumps({"answers": answers, "stats": instance.get_stats()}))
    finally:
        instance.close()
