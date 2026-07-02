from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from lmms_eval.agentic.types import EpisodeResult


class LoopSession(ABC):
    """Stateful rollout interface used by model-server schedulers."""

    @property
    @abstractmethod
    def done(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def next_request(self) -> Any | None:
        raise NotImplementedError

    @abstractmethod
    def apply_model_output(self, raw_output: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def result(self) -> EpisodeResult:
        raise NotImplementedError
