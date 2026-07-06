"""Composite verifier — chain multiple verifiers with fallback logic.

Typical use-case: try a cheap rule-based check first, fall back to an
expensive LLM judge only when the rule-based verifier is uncertain.
"""

from typing import Any, List

from .base import Verifier, VerifyResult


class CompositeVerifier(Verifier):
    """Try verifiers in sequence; stop at the first *confident* result.

    A verifier signals uncertainty by setting
    ``result.metadata["confident"] = False``.  When that happens the
    composite moves on to the next verifier in the chain.  The last
    verifier's result is always accepted regardless of confidence.

    Example::

        pipeline = CompositeVerifier([
            ExactMatchVerifier(),   # fast, always confident
            OpenAIVerifier(),       # expensive fallback
        ])
    """

    def __init__(self, verifiers: List[Verifier]):
        if not verifiers:
            raise ValueError("CompositeVerifier requires at least one verifier")
        self.verifiers = verifiers

    def verify(self, question: str, prediction: str, ground_truth: str, **kwargs: Any) -> VerifyResult:
        last_result: VerifyResult = VerifyResult(score=0.0, is_correct=False, raw_output="no_verifiers_ran")
        for v in self.verifiers:
            result = v.verify(question, prediction, ground_truth, **kwargs)
            last_result = result
            if result.metadata.get("confident", True):
                return result
        return last_result
