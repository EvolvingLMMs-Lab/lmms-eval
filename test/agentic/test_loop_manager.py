from __future__ import annotations

import threading

import pytest

from lmms_eval.agentic.loop.manager import LoopManager, RolloutJob
from lmms_eval.agentic.loop.worker import LoopWorker
from lmms_eval.agentic.types import EnvState, EpisodeResult


class _Worker(LoopWorker):
    def __init__(self, barrier: threading.Barrier | None = None) -> None:
        self.barrier = barrier

    def run(self, doc, seed=None, agent_id="agent"):
        if self.barrier is not None:
            self.barrier.wait(timeout=2)
        return EpisodeResult(
            final_state=EnvState(env_id=str(doc), step_idx=0, observation={}, terminal=True),
            metadata={"doc": doc, "seed": seed, "agent_id": agent_id},
        )


def test_loop_manager_runs_workers_in_threads_and_preserves_order():
    barrier = threading.Barrier(3)
    manager = LoopManager(worker=lambda: _Worker(barrier), max_workers=3)

    results = manager.run_many(["a", "b", "c"], seeds=[1, 2, 3], agent_ids=["x", "y", "z"])

    assert [result.final_state.env_id for result in results] == ["a", "b", "c"]
    assert [result.metadata["seed"] for result in results] == [1, 2, 3]
    assert [result.metadata["agent_id"] for result in results] == ["x", "y", "z"]


def test_loop_manager_rejects_shared_worker_for_parallel_run():
    manager = LoopManager(worker=_Worker(), max_workers=2)

    with pytest.raises(ValueError, match="worker factory"):
        manager.run_many(["a", "b"])


def test_loop_manager_runs_rollout_jobs_in_threads_and_preserves_order():
    barrier = threading.Barrier(3)

    def make_job(index):
        def run_serial(_model_server):
            barrier.wait(timeout=2)
            return f"job{index}"

        return RolloutJob(index=index, make_session=lambda _model_server: None, run_serial=run_serial)

    manager = LoopManager(max_workers=3)

    assert manager.run_jobs([make_job(index) for index in range(3)], model_server=object()) == ["job0", "job1", "job2"]
