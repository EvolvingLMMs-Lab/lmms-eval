"""Core abstractions for the verification pipeline.

A *Verifier* answers one question: is the model's prediction correct given
the ground truth?  An *Extractor* cleans/normalizes raw model output before
it reaches the verifier.

Both operate **per-sample** (single prediction string), which distinguishes
them from the batch-level ``Filter`` system in ``lmms_eval.filters``.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class VerifyResult:
    """Outcome of a single verification."""

    score: float  # 0.0–1.0 normalized
    is_correct: bool
    raw_output: str = ""  # judge's raw text (debugging / logging)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Verifier(ABC):
    """Abstract base for all verifiers."""

    @abstractmethod
    def verify(
        self,
        question: str,
        prediction: str,
        ground_truth: str,
        **kwargs: Any,
    ) -> VerifyResult: ...

    def __call__(self, question: str, prediction: str, ground_truth: str, **kwargs: Any) -> VerifyResult:
        return self.verify(question, prediction, ground_truth, **kwargs)


class Extractor(ABC):
    """Abstract base for per-sample text extractors.

    Extractors transform raw model output into a clean prediction string
    *before* it reaches the verifier.
    """

    @abstractmethod
    def extract(self, text: str, **kwargs: Any) -> str: ...

    def __call__(self, text: str, **kwargs: Any) -> str:
        return self.extract(text, **kwargs)


def parse_binary_response(text: str) -> bool:
    """Parse a judge response into True (correct) or False (incorrect).

    Handles common LLM judge formats: CORRECT/INCORRECT, 1/0, Yes/No.
    """
    upper = text.strip().upper()
    if "INCORRECT" in upper or "NOT_ATTEMPTED" in upper:
        return False
    if "CORRECT" in upper:
        return True
    lower = text.strip().lower()
    for sig in ("correct", "yes", "true", "1", "[1]"):
        if sig in lower:
            return True
    return False
