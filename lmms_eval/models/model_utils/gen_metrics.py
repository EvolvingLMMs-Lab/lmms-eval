import time
from numbers import Number
from typing import Any, Callable, Dict, List, Optional

from loguru import logger as eval_logger

_THROUGHPUT_METRICS_HISTORY: List[Dict[str, Any]] = []


def reset_logged_metrics() -> None:
    """Reset in-memory throughput metrics captured during a run."""

    _THROUGHPUT_METRICS_HISTORY.clear()


def get_logged_metrics_history() -> List[Dict[str, Any]]:
    """Return captured throughput metrics in collection order."""

    return list(_THROUGHPUT_METRICS_HISTORY)


def summarize_logged_metrics() -> Dict[str, Any]:
    """Aggregate captured throughput metrics for final reporting."""

    if not _THROUGHPUT_METRICS_HISTORY:
        return {}

    total_gen_tokens = 0.0
    total_elapsed_time = 0.0
    total_requests = 0.0
    avg_speed_vals: List[float] = []
    additional_numeric: Dict[str, List[float]] = {}

    for metric in _THROUGHPUT_METRICS_HISTORY:
        token_val = metric.get("total_gen_tokens")
        latency_val = metric.get("total_elapsed_time")
        speed_val = metric.get("avg_speed")
        requests_val = metric.get("total_requests")
        if requests_val is None:
            requests_val = metric.get("request_count")
        if requests_val is None:
            requests_val = metric.get("num_requests")

        if isinstance(token_val, Number):
            total_gen_tokens += float(token_val)
        if isinstance(latency_val, Number):
            total_elapsed_time += float(latency_val)
        if isinstance(speed_val, Number):
            avg_speed_vals.append(float(speed_val))
        if isinstance(requests_val, Number):
            total_requests += float(requests_val)

        for key, value in metric.items():
            if key in {"total_gen_tokens", "total_elapsed_time", "avg_speed", "total_requests", "request_count", "num_requests"}:
                continue
            if isinstance(value, Number):
                additional_numeric.setdefault(key, []).append(float(value))

    summary: Dict[str, Any] = {
        "total_gen_tokens": int(total_gen_tokens) if total_gen_tokens.is_integer() else total_gen_tokens,
        "total_elapsed_time": total_elapsed_time,
        "avg_speed": (total_gen_tokens / total_elapsed_time) if total_elapsed_time > 0 else (sum(avg_speed_vals) / len(avg_speed_vals) if avg_speed_vals else 0.0),
    }

    if total_requests > 0:
        summary["avg_latency"] = total_elapsed_time / total_requests

    for key, values in additional_numeric.items():
        if values:
            summary[f"avg_{key}"] = sum(values) / len(values)

    return summary


def _record_metrics(
    total_elapsed_time: float,
    total_gen_tokens: int,
    avg_speed: float,
    additional_metrics: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "total_elapsed_time": total_elapsed_time,
        "total_gen_tokens": total_gen_tokens,
        "avg_speed": avg_speed,
    }
    if additional_metrics:
        payload.update(additional_metrics)
    _THROUGHPUT_METRICS_HISTORY.append(payload)


def space_tokenizer(text: str) -> float:
    """
    A simple tokenizer that counts the token by the splitted space
    Then a rough estimate of the token count is returned. (times 1.5)
    Args:
        text (str): The input text to tokenize.
    """
    return len(text.split(" ")) * 1.5


def calculate_token_throughput(token_count: float, duration: float) -> float:
    """
    Calculate the token throughput.

    Args:
        token_count (float): The number of tokens processed.
        duration (float): The time taken to process the tokens in seconds.

    Returns:
        float: The token throughput in tokens per second.
    """
    if duration <= 0:
        return 0.0
    return token_count / duration


def log_metrics(
    total_elapsed_time: float,
    total_gen_tokens: int,
    avg_speed: float,
    additional_metrics: Optional[Dict[str, Any]] = None,
):
    """
    Log the metrics in a structured format.

    Args:
        total_elapsed_time (float): Sum of generation latencies in seconds.
        total_gen_tokens (int): The total number of generated tokens.
        avg_speed (float): The average speed in tokens per second.
        additional_metrics (Dict[str, Any]): Additional metrics to log.
    """
    required_stats = f"Metric summary - Total elapsed time: {total_elapsed_time:.3f}s, Total gen tokens: {total_gen_tokens}, Avg speed: {avg_speed:.1f} tokens/s"
    if additional_metrics is not None:
        required_stats += ", Additional metrics: "
        required_stats += ", ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in additional_metrics.items())
    eval_logger.info(required_stats)
    _record_metrics(
        total_elapsed_time=total_elapsed_time,
        total_gen_tokens=total_gen_tokens,
        avg_speed=avg_speed,
        additional_metrics=additional_metrics,
    )


class GenMetrics:
    """
    A class to manage the generation of metrics for model evaluation.
    """

    def __init__(self, tokenize_fn: Callable = space_tokenizer):
        self.tokenize_fn = tokenize_fn

    def __enter__(self):
        """
        Initialize the context manager.
        """
        self.metrics = {}
        self.start_time = time.perf_counter()
        return self

    def stop_timer(self):
        self.end_time = time.perf_counter()

    def log_metric(self, content: List[Any], additional_metrics: Optional[Dict[str, Any]] = None):
        num_tokens = sum(self.tokenize_fn(item) for item in content)
        duration = self.end_time - self.start_time
        throughput = calculate_token_throughput(num_tokens, duration)
        self.metrics = {
            "num_tokens": num_tokens,
            "duration": duration,
            "throughput": throughput,
        }
        if additional_metrics:
            self.metrics.update(additional_metrics)

        log_metrics(
            total_elapsed_time=duration,
            total_gen_tokens=int(num_tokens),
            avg_speed=throughput,
            additional_metrics=additional_metrics,
        )

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Finalize the context manager and return the collected metrics.
        """
        self.end_time = time.perf_counter()
        self.metrics["duration"] = self.end_time - self.start_time
        return self.metrics
