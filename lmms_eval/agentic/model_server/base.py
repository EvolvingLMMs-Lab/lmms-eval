from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

from lmms_eval.agentic.loop.session import LoopSession


@dataclass(slots=True)
class RolloutJob:
    """A rollout factory owned by the runner but scheduled by a ModelServer."""

    index: int
    make_session: Callable[["ModelServer"], LoopSession | None]
    run_serial: Callable[["ModelServer"], Any]


class AgentModel(ABC):
    """Minimal single-request model interface."""

    @abstractmethod
    def generate(self, request: Any) -> Any:
        raise NotImplementedError


class ModelServer(ABC):
    """Batch/scheduling boundary for agent inference."""

    @abstractmethod
    def generate(self, request: Any) -> Any:
        raise NotImplementedError

    def generate_batch(self, requests: list[Any]) -> list[Any]:
        return [self.generate(request) for request in requests]

    def run_rollouts(self, jobs: list[RolloutJob]) -> list[Any]:
        """Run rollout jobs through this server's scheduling policy.

        The default scheduler executes a bounded set of loop sessions in
        lock-step and calls ``generate_batch`` for all ready model requests.
        Backends with their own schedulers can override this method.
        """

        if not jobs:
            return []

        results_by_index: dict[int, Any] = {}
        batch_size = self.max_parallel_rollouts(len(jobs))
        for start in range(0, len(jobs), batch_size):
            chunk = jobs[start : start + batch_size]
            sessions: list[tuple[RolloutJob, LoopSession]] = []
            serial_jobs: list[RolloutJob] = []
            for job in chunk:
                session = job.make_session(self)
                if session is None:
                    serial_jobs.append(job)
                else:
                    sessions.append((job, session))

            if sessions:
                session_results = self.run_loop_sessions([session for _, session in sessions])
                for (job, _session), result in zip(sessions, session_results, strict=True):
                    results_by_index[job.index] = result

            for job in serial_jobs:
                results_by_index[job.index] = job.run_serial(self)

        return [results_by_index[job.index] for job in jobs]

    def run_loop_sessions(self, sessions: list[LoopSession]) -> list[Any]:
        while True:
            ready: list[tuple[LoopSession, Any]] = []
            active = False
            for session in sessions:
                if session.done:
                    continue
                active = True
                request = session.next_request()
                if request is not None:
                    ready.append((session, request))

            if not active:
                break
            if not ready:
                raise RuntimeError("Loop sessions are active but no model requests are ready")

            outputs = self.generate_batch([request for _session, request in ready])
            if len(outputs) != len(ready):
                raise RuntimeError(f"ModelServer returned {len(outputs)} outputs for {len(ready)} requests")
            for (session, _request), output in zip(ready, outputs, strict=True):
                session.apply_model_output(output)

        return [session.result() for session in sessions]

    def max_parallel_rollouts(self, requested: int) -> int:
        configured = getattr(self, "_max_parallel_rollouts", 1)
        if configured is None:
            return max(1, requested)
        return max(1, min(int(configured), requested))
