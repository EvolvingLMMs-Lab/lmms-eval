"""OpenAI / GPT-4o judge verifier.

Wraps the existing ``lmms_eval.llm_judge`` provider layer to expose a
``Verifier`` interface.  Two ``judge_type`` modes are supported:

* **binary** — correct / incorrect (uses ``evaluate_binary``)
* **comparative** — pairwise scoring (uses ``evaluate_comparative``)

Score-style output is controlled by *response_format* (``"binary"`` /
``"score"``) and full prompt control by *custom_prompt* — both independent
of *judge_type*.
"""

import logging
import os
from typing import Any, Callable, Optional, Tuple, Union

from .base import Verifier, VerifyResult, parse_binary_response

logger = logging.getLogger(__name__)


def _parse_score_response(text: str, score_range: Tuple[float, float] = (0.0, 10.0)) -> float:
    """Extract a numerical score and normalise to 0-1."""
    import re

    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not numbers:
        return 0.0
    raw = float(numbers[0])
    lo, hi = score_range
    if hi == lo:
        return 1.0 if raw >= hi else 0.0
    return max(0.0, min(1.0, (raw - lo) / (hi - lo)))


class OpenAIVerifier(Verifier):
    """Verification via OpenAI-compatible APIs (GPT-4o, Azure, vLLM, etc.).

    Parameters
    ----------
    model : str
        Model name, e.g. ``"gpt-4o-2024-11-20"``.
    api_type : str | None
        Provider key (``"openai"``, ``"azure"``, …).  Falls back to
        ``$API_TYPE`` env var.
    judge_type : str
        ``"binary"`` — correct/incorrect via ``evaluate_binary``.
        ``"comparative"`` — pairwise scoring via ``evaluate_comparative``.
        Score-style output is driven by *response_format* and custom
        prompts by *custom_prompt*, independent of *judge_type*.
    custom_prompt : str | callable | None
        For ``"custom"`` mode.  If a string, may contain
        ``{question}``, ``{prediction}``, ``{ground_truth}`` placeholders.
        If a callable, signature ``(question, prediction, ground_truth, **kw) -> str``.
    response_format : str
        How to parse the judge response: ``"binary"`` or ``"score"``.
    response_parser : callable | None
        Optional custom callable ``(str) -> VerifyResult`` that overrides
        *response_format*.  Useful when the judge returns task-specific
        formats (e.g. A/B/C letters).
    score_range : tuple[float, float]
        Min/max of the raw score the judge produces (used for normalisation).
    max_retries / retry_delay : int / float
        Retry parameters for transient API failures.

    Notes
    -----
    If the judge call fails (an exception is raised, or the provider reports
    ``success=False``), :meth:`verify` returns a :class:`VerifyResult` with
    ``metadata["judge_failed"] = True`` (and ``score=0.0`` /
    ``is_correct=False``) so callers can exclude infra failures instead of
    counting them as a genuinely wrong prediction.
    """

    def __init__(
        self,
        model: str = "gpt-4o-2024-11-20",
        api_type: Optional[str] = None,
        judge_type: str = "binary",
        custom_prompt: Optional[Union[str, Callable]] = None,
        response_format: str = "binary",
        response_parser: Optional[Callable] = None,
        score_range: Tuple[float, float] = (0.0, 10.0),
        max_retries: int = 5,
        retry_delay: float = 2.0,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        self.judge_type = judge_type
        self.custom_prompt = custom_prompt
        self.response_format = response_format
        self.response_parser = response_parser
        self.score_range = score_range
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        from lmms_eval.llm_judge import get_server
        from lmms_eval.llm_judge.protocol import ServerConfig

        api_type = api_type or os.getenv("API_TYPE", "openai")
        config = ServerConfig(
            model_name=model,
            temperature=temperature,
            max_tokens=max_tokens,
            num_retries=max_retries,
            retry_delay=retry_delay,
        )
        self._server = get_server(server_name=api_type, config=config)

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_prompt(self, question: str, prediction: str, ground_truth: str, **kwargs: Any) -> Optional[str]:
        """Build a prompt string for *custom* mode.  Returns ``None`` for
        *binary*/*score* modes (handled via ``evaluate_binary``)."""
        if self.custom_prompt is None:
            return None
        if callable(self.custom_prompt):
            return self.custom_prompt(question, prediction, ground_truth, **kwargs)
        return self.custom_prompt.format(
            question=question,
            prediction=prediction,
            ground_truth=ground_truth,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Core verify
    # ------------------------------------------------------------------

    def verify(self, question: str, prediction: str, ground_truth: str, **kwargs: Any) -> VerifyResult:
        try:
            return self._verify_once(question, prediction, ground_truth, **kwargs)
        except Exception as e:
            logger.error("OpenAI judge failed: %s", e)
            return VerifyResult(score=0.0, is_correct=False, raw_output=f"error: {e}", metadata={"judge_failed": True})

    def _verify_once(self, question: str, prediction: str, ground_truth: str, **kwargs: Any) -> VerifyResult:
        # --- custom prompt mode -------------------------------------------------
        prompt = self._build_prompt(question, prediction, ground_truth, **kwargs)
        if prompt is not None:
            from lmms_eval.llm_judge.protocol import Request

            request = Request(
                messages=[{"role": "user", "content": prompt}],
                images=kwargs.get("images"),
                config=self._server.config,
            )
            response = self._server.evaluate(request)
            return self._parse(response.content)

        # --- built-in binary mode -----------------------------------------------
        if self.judge_type == "binary":
            result = self._server.evaluate_binary(
                question=question,
                answer=ground_truth,
                prediction=prediction,
                output_format=kwargs.get("output_format", "0/1"),
            )
            if not result.get("success", True):
                return VerifyResult(
                    score=0.0,
                    is_correct=False,
                    raw_output=result.get("raw_response", ""),
                    metadata={"model": result.get("model", ""), "judge_failed": True},
                )
            is_correct = bool(result.get("result"))
            return VerifyResult(
                score=1.0 if is_correct else 0.0,
                is_correct=is_correct,
                raw_output=result.get("raw_response", ""),
                metadata={"model": result.get("model", "")},
            )

        # --- built-in comparative mode ------------------------------------------
        if self.judge_type == "comparative":
            result = self._server.evaluate_comparative(
                question=question,
                response1=prediction,
                response2=kwargs.get("reference_response", ground_truth),
                custom_prompt=None,
                images=kwargs.get("images"),
            )
            if not result.get("success", True):
                return VerifyResult(
                    score=0.0,
                    is_correct=False,
                    raw_output=result.get("raw_response", ""),
                    metadata={"model": result.get("model", ""), "judge_failed": True},
                )
            scores = result.get("scores", (-1.0, -1.0))
            s1, s2 = scores if isinstance(scores, tuple) else (-1.0, -1.0)
            lo, hi = self.score_range
            norm = max(0.0, min(1.0, (s1 - lo) / (hi - lo))) if hi > lo else 0.0
            return VerifyResult(
                score=norm,
                is_correct=s1 > s2,
                raw_output=result.get("raw_response", ""),
                metadata={"score_1": s1, "score_2": s2, "model": result.get("model", "")},
            )

        raise ValueError(f"Unknown judge_type: {self.judge_type!r}")

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse(self, text: str) -> VerifyResult:
        if self.response_parser is not None:
            return self.response_parser(text)
        if self.response_format == "binary":
            correct = parse_binary_response(text)
            return VerifyResult(score=1.0 if correct else 0.0, is_correct=correct, raw_output=text)
        if self.response_format == "score":
            score = _parse_score_response(text, self.score_range)
            return VerifyResult(score=score, is_correct=score >= 0.5, raw_output=text)
        raise ValueError(f"Unknown response_format: {self.response_format!r}")
