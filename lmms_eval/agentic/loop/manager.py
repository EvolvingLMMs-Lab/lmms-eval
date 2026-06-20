from __future__ import annotations

from typing import Any

from lmms_eval.agentic.loop.worker import LoopWorker
from lmms_eval.agentic.types import EpisodeResult


class LoopManager:
    """Placeholder for future parallel scheduling."""

    def __init__(self, worker: LoopWorker) -> None:
        self.worker = worker

    def run_many(self, docs: list[Any]) -> list[EpisodeResult]:
        return [self.worker.run(doc) for doc in docs]
