"""Verification pipeline — the glue between extractors and verifiers.

A ``VerificationPipeline`` chains zero-or-more **extractors** (per-sample
text transforms) followed by exactly one **verifier**.  This is the main
entry-point tasks use to plug verification into ``process_results``.

::

    raw model output ──► extractor₁ ──► extractor₂ ──► verifier ──► VerifyResult
"""

from typing import Any, Dict, List, Optional, Union

from .base import Extractor, Verifier, VerifyResult


class VerificationPipeline:
    """Extraction → Verification in a single callable.

    Parameters
    ----------
    extractors : list[Extractor]
        Ordered per-sample text transforms applied before verification.
    verifier : Verifier
        The verifier that produces the final ``VerifyResult``.

    Examples
    --------
    ::

        from lmms_eval.verifiers import VerificationPipeline
        from lmms_eval.verifiers.extractors import StripReasoningExtractor
        from lmms_eval.verifiers.openai import OpenAIVerifier

        pipeline = VerificationPipeline(
            extractors=[StripReasoningExtractor()],
            verifier=OpenAIVerifier(model="gpt-4o-2024-11-20"),
        )
        result = pipeline("What color is the sky?", "<think>hmm</think>Blue", "Blue")
        assert result.is_correct
    """

    def __init__(
        self,
        extractors: Optional[List[Extractor]] = None,
        verifier: Optional[Verifier] = None,
    ):
        self.extractors: List[Extractor] = extractors or []
        if verifier is None:
            raise ValueError("VerificationPipeline requires a verifier")
        self.verifier: Verifier = verifier

    def __call__(
        self,
        question: str,
        prediction: str,
        ground_truth: str,
        **kwargs: Any,
    ) -> VerifyResult:
        """Run the full pipeline: extract → verify."""
        cleaned = prediction
        for ext in self.extractors:
            cleaned = ext(cleaned, **kwargs)

        result = self.verifier.verify(question, cleaned, ground_truth, **kwargs)
        result.metadata["raw_prediction"] = prediction
        result.metadata["extracted_prediction"] = cleaned
        return result

    def verify(self, question: str, prediction: str, ground_truth: str, **kwargs: Any) -> VerifyResult:
        """Alias for ``__call__``."""
        return self(question, prediction, ground_truth, **kwargs)
