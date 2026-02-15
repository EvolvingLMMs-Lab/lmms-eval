import hashlib
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class AdaptiveConcurrencyConfig:
    min_concurrency: int
    max_concurrency: int
    target_latency_s: float
    increase_step: float
    decrease_factor: float
    failure_threshold: float

    @classmethod
    def from_raw(
        cls,
        min_concurrency: int,
        max_concurrency: int,
        target_latency_s: float,
        increase_step: float,
        decrease_factor: float,
        failure_threshold: float,
    ) -> "AdaptiveConcurrencyConfig":
        sanitized_min = max(1, int(min_concurrency))
        sanitized_max = max(sanitized_min, int(max_concurrency))
        return cls(
            min_concurrency=sanitized_min,
            max_concurrency=sanitized_max,
            target_latency_s=max(0.1, float(target_latency_s)),
            increase_step=max(0.01, float(increase_step)),
            decrease_factor=min(0.95, max(0.1, float(decrease_factor))),
            failure_threshold=min(1.0, max(0.0, float(failure_threshold))),
        )


@dataclass(frozen=True)
class AdaptiveConcurrencyDecision:
    current_concurrency: int
    next_concurrency: int
    failure_rate: float
    rate_limit_rate: float
    p95_latency_s: float
    should_reduce: bool


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def make_prefix_hash(text: str, prefix_chars: int = 256) -> str:
    clipped = (text or "")[: max(1, int(prefix_chars))]
    return hashlib.sha1(clipped.encode("utf-8", errors="ignore")).hexdigest()


def extract_text_prefix_from_chat_messages(messages: Sequence[Any], max_chars: int = 256) -> str:
    remaining = max(1, int(max_chars))
    chunks = []

    for message in messages or []:
        if remaining <= 0:
            break
        if not isinstance(message, dict):
            continue

        content = message.get("content")
        if isinstance(content, str):
            segment = content[:remaining]
            chunks.append(segment)
            remaining -= len(segment)
            continue

        if not isinstance(content, list):
            continue

        for item in content:
            if remaining <= 0:
                break
            if isinstance(item, str):
                segment = item[:remaining]
                chunks.append(segment)
                remaining -= len(segment)
                continue
            if not isinstance(item, dict):
                continue

            text_value = None
            if item.get("type") == "text":
                text_value = item.get("text")
            elif "text" in item:
                text_value = item.get("text")
            if isinstance(text_value, str):
                segment = text_value[:remaining]
                chunks.append(segment)
                remaining -= len(segment)

    return "".join(chunks)


def is_rate_limit_error(error_msg: str) -> bool:
    lowered = error_msg.lower()
    patterns = (
        "429",
        "rate limit",
        "too many requests",
        "quota",
        "rate_limit",
        "throttle",
        "throttled",
        "requests per min",
        "tokens per min",
        "rpm",
        "tpm",
    )
    return any(pattern in lowered for pattern in patterns)


def compute_p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int(0.95 * (len(sorted_values) - 1))
    return sorted_values[index]


def decide_next_concurrency(
    *,
    current_concurrency: int,
    total_requests: int,
    failed_requests: int,
    rate_limited_requests: int,
    latencies: Sequence[float],
    config: AdaptiveConcurrencyConfig,
) -> AdaptiveConcurrencyDecision:
    if total_requests <= 0:
        return AdaptiveConcurrencyDecision(
            current_concurrency=current_concurrency,
            next_concurrency=current_concurrency,
            failure_rate=0.0,
            rate_limit_rate=0.0,
            p95_latency_s=0.0,
            should_reduce=False,
        )

    failure_rate = failed_requests / total_requests
    rate_limit_rate = rate_limited_requests / total_requests
    p95_latency_s = compute_p95(latencies)

    rate_limit_reduce_threshold = max(0.02, config.failure_threshold)
    high_latency_threshold = config.target_latency_s * 1.1
    low_latency_threshold = config.target_latency_s * 0.85

    should_reduce = rate_limit_rate >= rate_limit_reduce_threshold or failure_rate > config.failure_threshold or (p95_latency_s > 0 and p95_latency_s > high_latency_threshold)
    should_increase = not should_reduce and rate_limit_rate == 0.0 and failure_rate <= (config.failure_threshold * 0.5) and (p95_latency_s == 0 or p95_latency_s < low_latency_threshold)

    if should_reduce:
        next_concurrency = max(
            config.min_concurrency,
            int(current_concurrency * config.decrease_factor),
        )
    elif should_increase:
        increase_delta = max(1, int(current_concurrency * config.increase_step))
        next_concurrency = min(
            config.max_concurrency,
            current_concurrency + increase_delta,
        )
    else:
        next_concurrency = current_concurrency

    next_concurrency = max(config.min_concurrency, next_concurrency)
    return AdaptiveConcurrencyDecision(
        current_concurrency=current_concurrency,
        next_concurrency=next_concurrency,
        failure_rate=failure_rate,
        rate_limit_rate=rate_limit_rate,
        p95_latency_s=p95_latency_s,
        should_reduce=should_reduce,
    )
