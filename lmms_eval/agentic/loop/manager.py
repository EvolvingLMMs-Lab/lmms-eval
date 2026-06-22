from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from lmms_eval.agentic.loop.worker import LoopWorker
from lmms_eval.agentic.types import EpisodeResult


class LoopManager:
    """Run rollout workers with optional thread-level parallelism.

    Use a worker factory when running with more than one thread. Most workers own
    mutable environment state, so sharing a single worker across concurrent
    rollouts is intentionally rejected.
    """

    def __init__(
        self,
        worker: LoopWorker | Callable[[], LoopWorker] | None = None,
        max_workers: int | str | None = 1,
    ) -> None:
        self.worker = worker
        self.max_workers = _normalize_max_workers(max_workers)

    def run_many(
        self,
        docs: Sequence[Any],
        *,
        seeds: Sequence[int | None] | None = None,
        agent_ids: Sequence[str] | None = None,
    ) -> list[EpisodeResult]:
        if self.worker is None:
            raise ValueError("LoopManager.run_many requires a worker or worker factory")
        if not docs:
            return []

        max_workers = min(self.max_workers, len(docs))
        if max_workers <= 1:
            return [self._run_one_doc(index, doc, seeds, agent_ids) for index, doc in enumerate(docs)]
        if not self._has_worker_factory:
            raise ValueError("LoopManager requires a worker factory when max_workers > 1")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(lambda item: self._run_one_doc(item[0], item[1], seeds, agent_ids), enumerate(docs)))

    def run_jobs(self, jobs: Sequence[Any], model_server: Any) -> list[Any]:
        """Run rollout jobs concurrently through their serial worker path.

        This is intended for OpenAI-compatible HTTP servers where request-level
        concurrency is handled by the server/client rather than by model-side
        tensor batching.
        """

        if not jobs:
            return []
        max_workers = min(self.max_workers, len(jobs))
        if max_workers <= 1:
            return [job.run_serial(model_server) for job in jobs]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(lambda job: job.run_serial(model_server), jobs))

    @property
    def _has_worker_factory(self) -> bool:
        return callable(self.worker) and not isinstance(self.worker, LoopWorker)

    def _build_worker(self) -> LoopWorker:
        if self._has_worker_factory:
            worker = self.worker()
            if not isinstance(worker, LoopWorker):
                raise TypeError("LoopManager worker factory must return a LoopWorker")
            return worker
        if isinstance(self.worker, LoopWorker):
            return self.worker
        raise TypeError("LoopManager worker must be a LoopWorker or a worker factory")

    def _run_one_doc(
        self,
        index: int,
        doc: Any,
        seeds: Sequence[int | None] | None,
        agent_ids: Sequence[str] | None,
    ) -> EpisodeResult:
        return self._build_worker().run(
            doc,
            seed=_sequence_value(seeds, index, None),
            agent_id=_sequence_value(agent_ids, index, "agent"),
        )


def _normalize_max_workers(value: int | str | None) -> int:
    if value is None:
        return 1
    return max(1, int(value))


def _sequence_value(values: Sequence[Any] | None, index: int, default: Any) -> Any:
    if values is None or index >= len(values):
        return default
    return values[index]
