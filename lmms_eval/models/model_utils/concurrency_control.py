from dataclasses import dataclass
from typing import Sequence


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


def is_rate_limit_error(error_msg: str) -> bool:
    lowered = error_msg.lower()
    return (
        "429" in lowered
        or "rate limit" in lowered
        or "too many requests" in lowered
        or "quota" in lowered
    )


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

    should_reduce = (
        rate_limit_rate > 0
        or failure_rate > config.failure_threshold
        or (p95_latency_s > 0 and p95_latency_s > config.target_latency_s)
    )

    if should_reduce:
        next_concurrency = max(
            config.min_concurrency,
            int(current_concurrency * config.decrease_factor),
        )
    else:
        increase_delta = max(1, int(current_concurrency * config.increase_step))
        next_concurrency = min(
            config.max_concurrency,
            current_concurrency + increase_delta,
        )

    next_concurrency = max(config.min_concurrency, next_concurrency)
    return AdaptiveConcurrencyDecision(
        current_concurrency=current_concurrency,
        next_concurrency=next_concurrency,
        failure_rate=failure_rate,
        rate_limit_rate=rate_limit_rate,
        p95_latency_s=p95_latency_s,
        should_reduce=should_reduce,
    )
