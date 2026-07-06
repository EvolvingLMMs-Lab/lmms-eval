"""Rule-based verifiers — no external API calls required."""

import math
import re
from typing import Any, Optional

from .base import Verifier, VerifyResult


class ExactMatchVerifier(Verifier):
    """Verify by exact string match after normalization."""

    def __init__(self, ignore_case: bool = True, strip: bool = True):
        self.ignore_case = ignore_case
        self.strip = strip

    def _normalize(self, text: str) -> str:
        if self.strip:
            text = text.strip()
        if self.ignore_case:
            text = text.lower()
        return text

    def verify(self, question: str, prediction: str, ground_truth: str, **kwargs: Any) -> VerifyResult:
        pred = self._normalize(prediction)
        gt = self._normalize(ground_truth)
        correct = pred == gt
        return VerifyResult(
            score=1.0 if correct else 0.0,
            is_correct=correct,
            raw_output=f"exact_match: '{pred}' vs '{gt}'",
            metadata={"confident": correct},
        )


class ContainsVerifier(Verifier):
    """Verify that the prediction contains the ground truth."""

    def __init__(self, ignore_case: bool = True):
        self.ignore_case = ignore_case

    def verify(self, question: str, prediction: str, ground_truth: str, **kwargs: Any) -> VerifyResult:
        pred = prediction.lower() if self.ignore_case else prediction
        gt = ground_truth.lower() if self.ignore_case else ground_truth
        correct = gt in pred
        return VerifyResult(
            score=1.0 if correct else 0.0,
            is_correct=correct,
            raw_output=f"contains: '{gt}' in prediction={correct}",
            metadata={"confident": correct},
        )


class MCQMatchVerifier(Verifier):
    """Verify MCQ answer by letter match.

    Handles common format variations: ``A``, ``(A)``, ``A.``, ``A)``.
    """

    def verify(self, question: str, prediction: str, ground_truth: str, **kwargs: Any) -> VerifyResult:
        pred = prediction.strip().upper().strip("(). ")
        gt = ground_truth.strip().upper().strip("(). ")
        # Parse-based: confident only when a single MCQ letter was extracted.
        # An unparseable prediction is low-confidence so a composite chain
        # falls through to a downstream (e.g. LLM) verifier.
        extracted = bool(re.fullmatch(r"[A-Z]", pred))
        correct = pred == gt
        return VerifyResult(
            score=1.0 if correct else 0.0,
            is_correct=correct,
            raw_output=f"mcq_match: '{pred}' vs '{gt}'",
            metadata={"confident": extracted},
        )


class NumericToleranceVerifier(Verifier):
    """Verify numerical answers within a tolerance."""

    def __init__(self, rel_tol: float = 1e-3, abs_tol: float = 1e-6):
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol

    @staticmethod
    def _parse_number(text: str) -> Optional[float]:
        text = text.strip().replace(",", "").replace("%", "")
        try:
            return float(text)
        except ValueError:
            numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
            return float(numbers[-1]) if numbers else None

    def verify(self, question: str, prediction: str, ground_truth: str, **kwargs: Any) -> VerifyResult:
        pred_num = self._parse_number(prediction)
        gt_num = self._parse_number(ground_truth)

        if pred_num is None or gt_num is None:
            return VerifyResult(
                score=0.0,
                is_correct=False,
                raw_output=f"numeric_parse_fail: pred={prediction!r} gt={ground_truth!r}",
                metadata={"confident": False},
            )

        correct = math.isclose(pred_num, gt_num, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
        # Parse-based: both numbers parsed cleanly, so this is a confident
        # verdict even when wrong (no fallback for a cleanly-parsed mismatch).
        return VerifyResult(
            score=1.0 if correct else 0.0,
            is_correct=correct,
            raw_output=f"numeric: {pred_num} vs {gt_num} (rel_tol={self.rel_tol})",
            metadata={"confident": True},
        )
