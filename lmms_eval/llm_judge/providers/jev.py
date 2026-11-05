"""TypeSafe Jev judge provider.

Jev is a System One model. It does not generate text. ``POST /v1/systemone``
returns a noul probability, a choice, or a score. This provider maps the
judge operations tasks already call onto those primitives and writes the
canonical string those tasks parse (``Yes`` / ``No``, ``1`` / ``0``, a score,
or the chosen option).

See ``docs/guides/llm_judge.md`` for which tasks can switch to ``API_TYPE=jev``.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import requests
from loguru import logger as eval_logger

from lmms_eval.models.model_utils.usage_metrics import log_usage

from ..base import ServerInterface
from ..protocol import Request, Response, ServerConfig
from ..utils import JudgePromptBuilder, ResponseParser

JEV_API_URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_JEV_MODEL = "jev-latest"
JEV_EMPTY_RESPONSE = "Jev returned an empty or unusable answer"
MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10

_BINARY_OUTPUT_FORMATS = {"0/1", "1/0", "yes/no", "true/false", "correct/incorrect", "label"}
_IMAGE_PART_TYPES = {"image", "image_url", "input_image"}
_CORRECTNESS_SCORE_RE = re.compile(r"correctness score is (.+?)(?:\)\.|\.$)", re.IGNORECASE | re.DOTALL)
_SCALE_RE = re.compile(
    r"(?:scale|score|from|between)\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*(?:to|and|-)\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_CJK_SCALE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*分\s*到\s*(\d+(?:\.\d+)?)\s*分")
_EXTRACTION_RE = re.compile(r"extracted answer\s*:", re.IGNORECASE)
_PAIR_BLOCK_RE = re.compile(r"\[\s*\w+\s+([12])\s*\]")


class JevAPIError(RuntimeError):
    """A non-retryable Jev HTTP error, or a retryable status that ran out of attempts."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        super().__init__(f"Jev API rejected the request with HTTP {status_code}: {detail}")


class JevTextOnlyError(ValueError):
    """Raised when a judge request includes an image. Jev reads text only."""

    def __init__(self) -> None:
        super().__init__("Jev is text-only and cannot judge prompts that include images. Use a vision-capable judge for this task.")


class JevUnsupportedPromptError(ValueError):
    """Raised when a prompt needs free-text generation instead of a Jev decision."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Jev does not generate text. {reason} Set judge_type to 'binary', 'score', or 'choice', or pass structured fields (question, answer, prediction, options).")


@dataclass
class _QuestionSpec:
    """One Jev question plus the scale used to adapt its answer."""

    kind: str
    instructions: str
    criteria: Any
    lo: float = 0.0
    hi: float = 1.0
    level_count: int = 2
    binary_style: str = "yes/no"
    threshold: float = 0.5


@dataclass
class _Plan:
    """The Jev request to send and how to turn the answers back into judge text."""

    state: Any
    questions: Dict[str, _QuestionSpec]
    adapter: str


def _default_config(config: Optional[ServerConfig]) -> ServerConfig:
    """Use jev-latest when the caller does not pass a config.

    ``ServerInterface`` otherwise fills in ``model_name="gpt-4"``, which the
    System One endpoint rejects.
    """
    if config is None:
        return ServerConfig(model_name=DEFAULT_JEV_MODEL)
    return config


_logged_model_fallback = False


def resolve_model_name(configured: str) -> str:
    """Return a Jev model id.

    Tasks usually put the generative judge name (``gpt-4o``) on
    ``ServerConfig`` via ``MODEL_VERSION``. Those names are not Jev ids.
    A name that already starts with ``jev-`` is sent as-is. Otherwise
    ``TYPESAFE_MODEL`` (then ``JEV_MODEL``) wins, and the default is
    ``jev-latest``.
    """
    global _logged_model_fallback
    if configured.startswith("jev-"):
        return configured
    override = os.getenv("TYPESAFE_MODEL") or os.getenv("JEV_MODEL")
    if override:
        return override
    if configured != DEFAULT_JEV_MODEL and not _logged_model_fallback:
        _logged_model_fallback = True
        eval_logger.warning(f"Jev expects a model id such as jev-latest or jev-1.14.0; using {DEFAULT_JEV_MODEL} instead of {configured!r}.")
    return DEFAULT_JEV_MODEL


def resolve_threshold(config: ServerConfig) -> float:
    """Yes/no cutoff. Config, then ``evaluation_criteria``, then the env, then 0.5."""
    if config.noul_threshold is not None:
        value = float(config.noul_threshold)
    elif config.evaluation_criteria and "noul_threshold" in config.evaluation_criteria:
        value = float(config.evaluation_criteria["noul_threshold"])
    else:
        env_value = os.getenv("JEV_NOUL_THRESHOLD")
        value = float(env_value) if env_value else 0.5
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Jev noul threshold must be between 0 and 1, got {value}")
    return value


def format_number(value: float) -> str:
    """Render a score the way task parsers read it: a bare number, no trailing zeros."""
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    if text in {"", "-", "-0"}:
        return "0"
    return text


def level_count(lo: float, hi: float) -> int:
    """How many ordered score levels cover ``[lo, hi]``.

    An inclusive integer lattice that fits in 2..10 levels is used as-is, so
    0..1 is the two endpoints and the continuous score lands between them.
    Wider ranges use 10 evenly spaced levels, the API maximum.
    """
    if hi < lo:
        lo, hi = hi, lo
    span = hi - lo
    if span == 0:
        raise ValueError("Jev score scale must have a non-zero range")
    if span.is_integer() and MIN_SCORE_LEVELS - 1 <= int(span) <= MAX_SCORE_LEVELS - 1:
        return int(span) + 1
    return MAX_SCORE_LEVELS


def level_labels(lo: float, hi: float, count: int) -> List[str]:
    """Ordered level descriptions. Index 0 is the low end of the task scale."""
    labels: List[str] = []
    for index in range(count):
        value = lo + (hi - lo) * index / (count - 1)
        if index == 0:
            place = "lowest"
        elif index == count - 1:
            place = "highest"
        else:
            place = "intermediate"
        labels.append(f"{format_number(value)} ({place})")
    return labels


def map_level_score(raw: float, lo: float, hi: float, levels: int) -> float:
    """Map a Jev level index (continuous, 0 .. levels-1) onto the task scale."""
    if levels <= 1:
        return lo
    bounded = min(max(raw, 0.0), float(levels - 1))
    return lo + (hi - lo) * (bounded / (levels - 1))


def detect_binary_style(text: str) -> Optional[str]:
    """Which canonical yes/no string the prompt tells the parser to expect."""
    lowered = text.lower()
    if "<label>true</label>" in lowered and "<label>false</label>" in lowered:
        return "label"
    if "correct_or_not" in lowered or re.search(r"\bcorrect or incorrect\b", lowered):
        return "correct/incorrect"
    if re.search(r"\btrue[\"']?\s+or\s+[\"']?false\b", lowered) or re.search(r'respond with ["\']true["\']', lowered):
        return "true/false"
    if re.search(r"\byes[\"']?\s+or\s+[\"']?no\b", lowered) or re.search(r'return only ["\']?yes["\']?', lowered) or re.search(r'only output a single ["\']?yes["\']?', lowered):
        return "yes/no"
    if re.search(r"\b1\s+or\s+0\b", lowered) or re.search(r'return only ["\']?1["\']?\s+or\s+["\']?0', lowered):
        return "0/1"
    return None


def detect_score_range(text: str) -> Optional[Tuple[float, float]]:
    """Pull a numeric scale out of a judge prompt, if it states one."""
    match = _CORRECTNESS_SCORE_RE.search(text)
    if match:
        numbers = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", match.group(1))]
        if len(numbers) >= 2:
            return min(numbers), max(numbers)
    match = _SCALE_RE.search(text)
    if match:
        return float(match.group(1)), float(match.group(2))
    match = _CJK_SCALE_RE.search(text)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None


def binary_content(positive: bool, style: str) -> str:
    """Canonical text for a noul decision. Existing parsers read these strings."""
    if style in {"0/1", "1/0"}:
        return "1" if positive else "0"
    if style == "true/false":
        return "True" if positive else "False"
    if style == "correct/incorrect":
        return "Correct" if positive else "Incorrect"
    if style == "label":
        return "<label>True</label>" if positive else "<label>False</label>"
    return "Yes" if positive else "No"


def _is_image_part(part: Mapping[str, Any]) -> bool:
    kind = str(part.get("type", "")).lower()
    return kind in _IMAGE_PART_TYPES or "image_url" in part


def _content_text(content: Any) -> str:
    """Flatten message content to text. Image parts are a hard error."""
    if isinstance(content, str):
        if "data:image/" in content:
            raise JevTextOnlyError()
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, str):
                if "data:image/" in part:
                    raise JevTextOnlyError()
                parts.append(part)
                continue
            if not isinstance(part, dict):
                continue
            if _is_image_part(part):
                raise JevTextOnlyError()
            text = part.get("text")
            if isinstance(text, str):
                if "data:image/" in text:
                    raise JevTextOnlyError()
                parts.append(text)
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def message_text(messages: Optional[Sequence[Mapping[str, Any]]]) -> str:
    """Join system and user text. Image content raises ``JevTextOnlyError``."""
    chunks: List[str] = []
    for message in messages or []:
        if not isinstance(message, Mapping):
            continue
        text = _content_text(message.get("content"))
        if text:
            chunks.append(text)
    return "\n\n".join(chunks)


def _build_state(request: Request, text: str) -> Any:
    """Prefer a structured state object. A prompt-only request stays a string."""
    state: Dict[str, Any] = {}
    if request.question:
        state["question"] = request.question
    if request.answer is not None:
        state["reference_answer"] = str(request.answer)
    if request.prediction is not None:
        state["prediction"] = str(request.prediction)
    if request.context:
        state["context"] = request.context
    if request.response1 is not None:
        state["response_1"] = request.response1
    if request.response2 is not None:
        state["response_2"] = request.response2
    if request.options:
        state["options"] = [str(option) for option in request.options]
    if text:
        state["judge_prompt"] = text
    if not state:
        return ""
    if set(state) == {"judge_prompt"}:
        return text
    return state


def _noul_spec(style: str, threshold: float) -> _QuestionSpec:
    return _QuestionSpec(
        kind="noul",
        instructions=("Is the prediction correct for the question, given the reference answer and the judge prompt? A yes means the positive criteria are met."),
        criteria={
            "true": "The judgment is positive (yes, correct, true, or 1).",
            "false": "The judgment is negative (no, incorrect, false, or 0).",
        },
        binary_style=style,
        threshold=threshold,
    )


def _score_spec(instructions: str, lo: float, hi: float, criteria: Optional[Sequence[str]] = None) -> _QuestionSpec:
    if criteria is None:
        count = level_count(lo, hi)
        labels = level_labels(lo, hi, count)
    else:
        labels = [str(item) for item in criteria]
        count = len(labels)
        if not MIN_SCORE_LEVELS <= count <= MAX_SCORE_LEVELS:
            raise ValueError(f"Jev score criteria need {MIN_SCORE_LEVELS} to {MAX_SCORE_LEVELS} levels, got {count}")
        # The descriptions are already the scale, so the level index is the score.
        lo, hi = 0.0, float(count - 1)
    return _QuestionSpec(
        kind="score",
        instructions=instructions,
        criteria=labels,
        lo=lo,
        hi=hi,
        level_count=count,
    )


def _choice_criteria(options: Sequence[Any], descriptions: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    if not options:
        raise JevUnsupportedPromptError("A choice judgment needs at least one option.")
    if len(options) > MAX_CHOICE_OPTIONS:
        raise ValueError(f"Jev choice questions accept at most {MAX_CHOICE_OPTIONS} options, got {len(options)}.")
    criteria: Dict[str, str] = {}
    for option in options:
        key = str(option)
        described = descriptions.get(key) if descriptions else None
        criteria[key] = key if described is None else str(described)
    return criteria


def _looks_comparative(text: str) -> bool:
    marks = set(_PAIR_BLOCK_RE.findall(text))
    lowered = text.lower()
    has_pair = marks >= {"1", "2"} or ("assistant 1" in lowered and "assistant 2" in lowered)
    return has_pair and detect_score_range(text) is not None


def _binary_style(config: ServerConfig, text: str) -> Optional[str]:
    if config.output_format in _BINARY_OUTPUT_FORMATS:
        return str(config.output_format)
    if config.judge_type == "binary":
        return detect_binary_style(text) or "yes/no"
    return None


def plan_request(request: Request, config: ServerConfig) -> _Plan:
    """Choose noul, score, or choice from the request. Images fail closed."""
    if request.images:
        raise JevTextOnlyError()
    text = message_text(request.messages)
    state = _build_state(request, text)
    threshold = resolve_threshold(config)
    explicit_binary = _binary_style(config, text)

    if explicit_binary is not None:
        return _Plan(state=state, questions={"judgment": _noul_spec(explicit_binary, threshold)}, adapter="noul")

    if config.judge_type == "choice" or request.options:
        descriptions = config.evaluation_criteria if isinstance(config.evaluation_criteria, Mapping) else None
        options = request.options or list((descriptions or {}).keys())
        return _Plan(
            state=state,
            questions={
                "judgment": _QuestionSpec(
                    kind="choice",
                    instructions="Which option is the right judgment?",
                    criteria=_choice_criteria(options, descriptions),
                )
            },
            adapter="choice",
        )

    if config.judge_type == "score":
        lo, hi = config.score_range or detect_score_range(text) or (0.0, 1.0)
        return _Plan(
            state=state,
            questions={"judgment": _score_spec("Rate the prediction on the requested scale.", float(lo), float(hi))},
            adapter="score",
        )

    paired = request.response1 is not None and request.response2 is not None
    if config.judge_type == "comparative" or paired or _looks_comparative(text):
        lo, hi = detect_score_range(text) or config.score_range or (1.0, 10.0)
        return _Plan(
            state=state,
            questions={
                "response_1": _score_spec("Rate response_1. The lowest level is the low end of the scale.", float(lo), float(hi)),
                "response_2": _score_spec("Rate response_2. The lowest level is the low end of the scale.", float(lo), float(hi)),
            },
            adapter="comparative",
        )

    if _EXTRACTION_RE.search(text):
        raise JevUnsupportedPromptError("This prompt asks for an extracted answer.")

    detected_binary = detect_binary_style(text)
    if detected_binary is not None:
        return _Plan(state=state, questions={"judgment": _noul_spec(detected_binary, threshold)}, adapter="noul")

    detected_range = detect_score_range(text) or config.score_range
    if detected_range is not None:
        lo, hi = detected_range
        return _Plan(
            state=state,
            questions={"judgment": _score_spec("Rate the prediction on the scale stated in the judge prompt.", float(lo), float(hi))},
            adapter="score",
        )

    if request.question and (request.answer is not None or request.prediction is not None):
        return _Plan(state=state, questions={"judgment": _noul_spec("yes/no", threshold)}, adapter="noul")

    raise JevUnsupportedPromptError("The prompt is not a yes/no decision, a numeric score, or a choice among options.")


def plan_rubric(question: str, prediction: str, rubric: Mapping[str, Any], config: ServerConfig, score_range: Optional[Tuple[float, float]] = None) -> _Plan:
    """One score question per rubric item. A list value is the level descriptions."""
    if not rubric:
        raise ValueError("Jev rubric evaluation needs at least one rubric item")
    default_range = score_range or config.score_range or (0.0, 1.0)
    questions: Dict[str, _QuestionSpec] = {}
    for key, value in rubric.items():
        name = str(key)
        if isinstance(value, (list, tuple)):
            questions[name] = _score_spec(f"Score `{name}` using the listed levels, from lowest to highest.", 0.0, 1.0, criteria=[str(item) for item in value])
        else:
            lo, hi = default_range
            questions[name] = _score_spec(
                f"Score `{name}` ({value}) from the low end of the scale to the high end.",
                float(lo),
                float(hi),
            )
    state = {"question": question, "prediction": prediction}
    return _Plan(state=state, questions=questions, adapter="rubric")


def build_payload(plan: _Plan, model_name: str) -> Dict[str, Any]:
    """Request body for ``POST /v1/systemone``."""
    questions: Dict[str, Any] = {}
    for key, spec in plan.questions.items():
        questions[key] = {"type": spec.kind, "instructions": spec.instructions, "criteria": spec.criteria}
    return {"model": model_name, "state": plan.state, "questions": questions}


def _as_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(JEV_EMPTY_RESPONSE)
    return float(value)


def _answer_for(answers: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    answer = answers.get(key)
    if not isinstance(answer, dict):
        raise ValueError(JEV_EMPTY_RESPONSE)
    return answer


def _confidence(answer: Mapping[str, Any]) -> Optional[float]:
    value = answer.get("confidence")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _require_answers(body: Any) -> Mapping[str, Any]:
    if not isinstance(body, dict):
        raise ValueError(JEV_EMPTY_RESPONSE)
    answers = body.get("answers")
    if not isinstance(answers, dict) or not answers:
        raise ValueError(JEV_EMPTY_RESPONSE)
    return answers


def adapt_response(body: Mapping[str, Any], plan: _Plan, requested_model: str) -> Response:
    """Turn Jev answers into the string the calling task already parses.

    A missing or blank decision raises ``ValueError`` with
    ``JEV_EMPTY_RESPONSE`` before any ``str.strip`` on ``None``.
    """
    answers = _require_answers(body)
    parsed: Dict[str, Any]
    if plan.adapter == "noul":
        spec = plan.questions["judgment"]
        probability = _as_number(_answer_for(answers, "judgment").get("noul"))
        positive = probability >= spec.threshold
        content = binary_content(positive, spec.binary_style)
        parsed = {"type": "noul", "noul": probability, "probability": probability, "threshold": spec.threshold, "decision": positive}
    elif plan.adapter == "score":
        spec = plan.questions["judgment"]
        answer = _answer_for(answers, "judgment")
        raw = _as_number(answer.get("score"))
        score = map_level_score(raw, spec.lo, spec.hi, spec.level_count)
        content = format_number(score)
        parsed = {"type": "score", "score": score, "raw_score": raw, "confidence": _confidence(answer), "legend": answer.get("legend"), "probabilities": answer.get("probabilities")}
    elif plan.adapter == "choice":
        answer = _answer_for(answers, "judgment")
        choice = answer.get("choice")
        if not isinstance(choice, str) or choice.strip() == "":
            raise ValueError(JEV_EMPTY_RESPONSE)
        content = choice
        probabilities = answer.get("probabilities") if isinstance(answer.get("probabilities"), dict) else {}
        parsed = {"type": "choice", "choice": choice, "probabilities": probabilities, "confidence": _confidence(answer)}
    elif plan.adapter == "comparative":
        rendered: List[str] = []
        parsed_scores: Dict[str, float] = {}
        for key, spec in plan.questions.items():
            raw = _as_number(_answer_for(answers, key).get("score"))
            score = map_level_score(raw, spec.lo, spec.hi, spec.level_count)
            parsed_scores[key] = score
            rendered.append(format_number(score))
        content = " ".join(rendered)
        parsed = {"type": "comparative", "scores": parsed_scores}
    elif plan.adapter == "rubric":
        scores: Dict[str, float] = {}
        for key, spec in plan.questions.items():
            raw = _as_number(_answer_for(answers, key).get("score"))
            scores[key] = map_level_score(raw, spec.lo, spec.hi, spec.level_count)
        content = json.dumps(scores)
        parsed = {"type": "rubric", "scores": scores}
    else:
        raise ValueError(f"Unknown Jev adapter {plan.adapter!r}")

    if not isinstance(content, str) or content.strip() == "":
        raise ValueError(JEV_EMPTY_RESPONSE)

    usage = _usage_dict(body.get("usage"))
    model_used = body.get("model") if isinstance(body.get("model"), str) and body.get("model") else requested_model
    if usage is not None:
        log_usage(
            model_name=model_used,
            task_name=None,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            reasoning_tokens=0,
            source="judge",
        )
    return Response(content=content, model_used=model_used, usage=usage, raw_response=dict(body), parsed_result=parsed, success=True)


def _usage_dict(usage: Any) -> Optional[Dict[str, int]]:
    if not isinstance(usage, dict):
        return None
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
    }


def _delay_seconds(attempt: int, base: float, headers: Optional[Mapping[str, Any]]) -> float:
    """Retry-After wins when it is a number of seconds. Otherwise exponential backoff."""
    if headers:
        raw = headers.get("Retry-After", headers.get("retry-after"))
        if raw is not None:
            try:
                return max(0.0, float(raw))
            except (TypeError, ValueError):
                pass
    return max(0.0, base) * (2**attempt)


def _body_snippet(response: requests.Response) -> str:
    text = response.text if isinstance(getattr(response, "text", None), str) else ""
    text = " ".join(text.split())
    if len(text) > 400:
        return text[:400] + "..."
    return text or f"HTTP {response.status_code}"


def _retryable_status(status: int) -> bool:
    return status == 429 or status == 529 or status >= 500


class JevProvider(ServerInterface):
    """TypeSafe Jev implementation of the judge interface.

    Environment variables
    ---------------------
    TYPESAFE_API_KEY : str
        Bearer token for ``https://api.typesafe.ai``. ``API_TYPE=jev`` or
        ``API_TYPE=typesafe`` selects this provider.
    JEV_NOUL_THRESHOLD : str, optional
        Yes/no cutoff in ``[0, 1]``. Default 0.5. A noul probability at or
        above the cutoff is the positive label. ``ServerConfig.noul_threshold``
        overrides this.
    TYPESAFE_API_URL : str, optional
        Full endpoint URL. Default ``JEV_API_URL``. Set it to
        ``https://openrouter.ai/api/v1/systemone`` to reach Jev through
        OpenRouter, with the OpenRouter key in ``TYPESAFE_API_KEY``.
    TYPESAFE_MODEL : str, optional
        Jev model id used when the task config still names a generative model.
        ``JEV_MODEL`` is the fallback. Default ``jev-latest``.
    """

    def __init__(self, config: Optional[ServerConfig] = None):
        super().__init__(_default_config(config))
        self.api_key = os.getenv("TYPESAFE_API_KEY", "")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def evaluate(self, request: Request) -> Response:
        """Evaluate one judgment. A single failure raises; the provider stays usable."""
        if not self.is_available():
            raise ValueError("TypeSafe API key not configured (set TYPESAFE_API_KEY)")
        config = request.config or self.config
        plan = plan_request(request, config)
        model_name = resolve_model_name(config.model_name)
        payload = build_payload(plan, model_name)
        return self._post_and_adapt(payload, config, plan, model_name)

    def evaluate_binary(self, question: str, answer: str, prediction: str, output_format: str = "0/1", custom_prompt: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Binary correctness via noul. ``probability`` is the raw yes-probability."""
        prompt = JudgePromptBuilder.build_binary_prompt(question=question, answer=answer, prediction=prediction, output_format=output_format, custom_prompt=custom_prompt, **kwargs)
        config = replace(self.config, judge_type="binary", output_format=output_format)
        request = Request(
            messages=[{"role": "user", "content": prompt}],
            question=question,
            answer=answer,
            prediction=prediction,
            config=config,
        )
        response = self.evaluate(request)
        details = response.parsed_result if isinstance(response.parsed_result, dict) else {}
        decision = details.get("decision")
        if isinstance(decision, bool):
            # The noul decision is authoritative: ResponseParser only knows 0/1 and yes/no.
            parsed_result = int(decision) if output_format in {"0/1", "1/0"} else decision
        else:
            parsed_result = ResponseParser.parse_binary_response(response.content, output_format)
        return {
            "result": parsed_result,
            "raw_response": response.content,
            "model": response.model_used,
            "prompt": prompt,
            "success": response.success,
            "probability": details.get("probability"),
            "threshold": details.get("threshold"),
        }

    def evaluate_with_rubric(self, question: str, prediction: str, rubric: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Score each rubric item. A list value is that item's ordered levels."""
        if not self.is_available():
            raise ValueError("TypeSafe API key not configured (set TYPESAFE_API_KEY)")
        score_range = kwargs.get("score_range")
        plan = plan_rubric(question, prediction, rubric, self.config, score_range=score_range)
        model_name = resolve_model_name(self.config.model_name)
        payload = build_payload(plan, model_name)
        response = self._post_and_adapt(payload, self.config, plan, model_name)
        scores = response.parsed_result.get("scores", {}) if isinstance(response.parsed_result, dict) else {}
        return {
            "scores": scores,
            "raw_response": response.content,
            "model": response.model_used,
            "prompt": "Jev score questions, one per rubric item.",
            "success": response.success,
        }

    def _post_and_adapt(self, payload: Dict[str, Any], config: ServerConfig, plan: _Plan, model_name: str) -> Response:
        """POST with backoff on 429/529/5xx and timeouts. 4xx fails immediately.

        Empty answer payloads raise ``ValueError`` and are retried, matching
        the chat-completion empty-response guard: nothing calls ``strip`` on
        a missing decision.
        """
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        attempts = max(1, config.num_retries)
        last_error: Optional[BaseException] = None
        for attempt in range(attempts):
            try:
                response = requests.post(os.getenv("TYPESAFE_API_URL") or JEV_API_URL, headers=headers, json=payload, timeout=config.timeout)
            except requests.exceptions.RequestException as exc:
                last_error = exc
                if attempt >= attempts - 1:
                    raise
                self._sleep(attempt, config, None, exc)
                continue

            status = response.status_code
            if _retryable_status(status):
                error = JevAPIError(status, _body_snippet(response))
                last_error = error
                if attempt >= attempts - 1:
                    raise error
                self._sleep(attempt, config, response.headers, error)
                continue
            if status >= 400:
                raise JevAPIError(status, _body_snippet(response))

            try:
                body = response.json()
            except ValueError as exc:
                last_error = ValueError(JEV_EMPTY_RESPONSE)
                if attempt >= attempts - 1:
                    raise last_error from exc
                self._sleep(attempt, config, None, last_error)
                continue

            try:
                return adapt_response(body, plan, model_name)
            except ValueError as exc:
                if JEV_EMPTY_RESPONSE not in str(exc):
                    raise
                last_error = exc
                if attempt >= attempts - 1:
                    raise
                self._sleep(attempt, config, None, exc)
                continue

        if last_error is not None:
            raise last_error
        raise ValueError(JEV_EMPTY_RESPONSE)

    def _sleep(self, attempt: int, config: ServerConfig, headers: Optional[Mapping[str, Any]], error: BaseException) -> None:
        delay = _delay_seconds(attempt, config.retry_delay, headers)
        eval_logger.warning(f"Jev attempt {attempt + 1}/{config.num_retries} failed: {error}. Retrying in {delay}s.")
        time.sleep(delay)


__all__ = [
    "DEFAULT_JEV_MODEL",
    "JEV_API_URL",
    "JEV_EMPTY_RESPONSE",
    "JevAPIError",
    "JevProvider",
    "JevTextOnlyError",
    "JevUnsupportedPromptError",
]
