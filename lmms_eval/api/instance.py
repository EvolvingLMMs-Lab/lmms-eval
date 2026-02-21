from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union


@dataclass
class TokenCounts:
    """Per-request token usage counters.

    Fields are ``None`` when the backend cannot report them (e.g. cached
    responses, or backends that only expose aggregate metrics).
    """

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None

    def to_dict(self) -> Dict[str, Optional[int]]:
        d: Dict[str, Optional[int]] = {}
        if self.input_tokens is not None:
            d["input_tokens"] = self.input_tokens
        if self.output_tokens is not None:
            d["output_tokens"] = self.output_tokens
        if self.reasoning_tokens is not None:
            d["reasoning_tokens"] = self.reasoning_tokens
        return d


@dataclass
class GenerationResult:
    """Typed wrapper returned by ``generate_until`` implementations.

    Models that wish to report per-sample token counts should return a
    list of ``GenerationResult`` instead of plain strings.  The evaluator
    transparently handles both ``str`` and ``GenerationResult`` outputs.
    """

    text: str
    token_counts: Optional[TokenCounts] = None


GenerationOutput = Union[str, GenerationResult]


def unwrap_generation_output(output: Any) -> Tuple[str, Optional[TokenCounts]]:
    """Normalize a model output into ``(text, token_counts | None)``.

    Accepts ``str``, ``GenerationResult``, or ``(str, dict)`` tuples for
    maximum backward compatibility.
    """
    if isinstance(output, GenerationResult):
        return output.text, output.token_counts
    if isinstance(output, str):
        return output, None

    if isinstance(output, (tuple, list)) and len(output) == 2 and isinstance(output[0], str):
        text, meta = output
        if isinstance(meta, TokenCounts):
            return text, meta
        if isinstance(meta, dict):
            return text, TokenCounts(
                input_tokens=meta.get("input_tokens"),
                output_tokens=meta.get("output_tokens"),
                reasoning_tokens=meta.get("reasoning_tokens"),
            )

    return str(output), None


@dataclass
class Instance:
    request_type: Literal["loglikelihood", "generate_until", "generate_until_multi_round", "generate_until_agentic"]
    arguments: tuple
    idx: int
    metadata: Dict[str, Union[str, int]] = field(default_factory=dict)
    resps: list = field(default_factory=list)
    filtered_resps: dict = field(default_factory=dict)
    raw_filtered_resps: dict = field(default_factory=dict)

    token_counts: List[Optional[TokenCounts]] = field(default_factory=list)

    # initialized after init
    task_name: str = None
    doc_id: str = None
    repeats: str = None
    doc: dict = None

    def __post_init__(self) -> None:
        # unpack metadata field
        self.task_name, self.doc_id, self.repeats = self.metadata["task"], self.metadata["doc_id"], self.metadata["repeats"]

    @property
    def args(self):
        """
        Returns (string,) where `string` is the string to calculate loglikelihood over
        """
        return self.arguments if isinstance(self.arguments, tuple) else (self.arguments,)
