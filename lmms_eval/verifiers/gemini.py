"""Google Gemini judge verifier.

Wraps the ``google-genai`` SDK to provide a ``Verifier`` interface.
Supports text-only and multimodal (text + image) evaluation.
"""

import io
import json
import logging
import os
import re
import time
from typing import Any, Callable, Optional, Tuple, Union

from .base import Verifier, VerifyResult, parse_binary_response

logger = logging.getLogger(__name__)


def _safe_parse_json(text: str) -> Optional[dict]:
    """Try to extract and parse JSON from a Gemini response."""
    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = re.sub(r"```", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None


class GeminiVerifier(Verifier):
    """Verification via Google Gemini (text and multimodal).

    Parameters
    ----------
    model : str
        Model name, e.g. ``"gemini-2.5-pro"``.
    api_key : str | None
        Google API key.  Falls back to ``$GOOGLE_API_KEY``.
    custom_prompt : str | callable | None
        Prompt template (with ``{question}``, ``{prediction}``,
        ``{ground_truth}`` placeholders) or a callable.
    response_format : str
        ``"binary"`` — parse Correct/Incorrect.
        ``"score"``  — extract numerical score and normalise.
        ``"json"``   — parse JSON and look for *score* key.
    score_range : tuple[float, float]
        Raw score range for normalisation.
    score_key : str
        Key to look up in JSON responses (default ``"score"``).
    max_retries / retry_delay : int / float
        Retry parameters.

    Notes
    -----
    If every retry is exhausted, :meth:`verify` returns a
    :class:`VerifyResult` with ``metadata["judge_failed"] = True`` (and
    ``score=0.0`` / ``is_correct=False``) so callers can exclude infra
    failures instead of counting them as a genuinely wrong prediction.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        api_key: Optional[str] = None,
        custom_prompt: Optional[Union[str, Callable]] = None,
        response_format: str = "binary",
        score_range: Tuple[float, float] = (0.0, 10.0),
        score_key: str = "score",
        max_retries: int = 5,
        retry_delay: float = 2.0,
        temperature: float = 0.0,
    ):
        self.model = model
        self.custom_prompt = custom_prompt
        self.response_format = response_format
        self.score_range = score_range
        self.score_key = score_key
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.temperature = temperature

        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self._api_key:
            raise RuntimeError("GOOGLE_API_KEY is required for GeminiVerifier")

        self._client = None  # lazy init

    def _get_client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_contents(self, question: str, prediction: str, ground_truth: str, **kwargs: Any) -> list:
        """Build a Gemini ``contents`` list (text + optional images)."""
        if self.custom_prompt is not None:
            if callable(self.custom_prompt):
                text = self.custom_prompt(question, prediction, ground_truth, **kwargs)
            else:
                text = self.custom_prompt.format(question=question, prediction=prediction, ground_truth=ground_truth, **kwargs)
        else:
            text = f"Question: {question}\n\n" f"Ground truth answer: {ground_truth}\n\n" f"Model prediction: {prediction}\n\n" "Is the model's prediction correct? Answer with CORRECT or INCORRECT."

        contents: list = []

        images = kwargs.get("images")
        if images:
            from google.genai import types
            from PIL import Image

            for img in images:
                if isinstance(img, bytes):
                    contents.append(types.Part.from_bytes(data=img, mime_type="image/png"))
                elif isinstance(img, str) and os.path.isfile(img):
                    with open(img, "rb") as f:
                        contents.append(types.Part.from_bytes(data=f.read(), mime_type="image/png"))
                elif isinstance(img, Image.Image):
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    contents.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"))
                else:
                    logger.warning("GeminiVerifier: unhandled image type %s; skipping", type(img).__name__)

        contents.append(text)
        return contents

    # ------------------------------------------------------------------
    # Core verify
    # ------------------------------------------------------------------

    def verify(self, question: str, prediction: str, ground_truth: str, **kwargs: Any) -> VerifyResult:
        contents = self._build_contents(question, prediction, ground_truth, **kwargs)

        for attempt in range(self.max_retries):
            try:
                client = self._get_client()
                response = client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config={"temperature": self.temperature},
                )
                return self._parse(response.text)
            except Exception as e:
                logger.warning("Gemini judge attempt %d/%d failed: %s", attempt + 1, self.max_retries, e)
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2**attempt))

        logger.error("Gemini judge failed after %d retries", self.max_retries)
        return VerifyResult(score=0.0, is_correct=False, raw_output="all_retries_exhausted", metadata={"judge_failed": True})

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse(self, text: str) -> VerifyResult:
        if self.response_format == "binary":
            correct = parse_binary_response(text)
            return VerifyResult(score=1.0 if correct else 0.0, is_correct=correct, raw_output=text)

        if self.response_format == "score":
            numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
            if numbers:
                raw = float(numbers[0])
                lo, hi = self.score_range
                norm = max(0.0, min(1.0, (raw - lo) / (hi - lo))) if hi > lo else 0.0
                return VerifyResult(score=norm, is_correct=norm >= 0.5, raw_output=text)
            return VerifyResult(score=0.0, is_correct=False, raw_output=text)

        if self.response_format == "json":
            parsed = _safe_parse_json(text)
            if parsed and self.score_key in parsed:
                raw = float(parsed[self.score_key])
                lo, hi = self.score_range
                norm = max(0.0, min(1.0, (raw - lo) / (hi - lo))) if hi > lo else 0.0
                return VerifyResult(
                    score=norm,
                    is_correct=norm >= 0.5,
                    raw_output=text,
                    metadata=parsed,
                )
            return VerifyResult(score=0.0, is_correct=False, raw_output=text, metadata={"parse_error": True})

        raise ValueError(f"Unknown response_format: {self.response_format!r}")
