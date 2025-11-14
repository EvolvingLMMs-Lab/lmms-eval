import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from loguru import logger as eval_logger


def space_tokenizer(text: str) -> int:
    """
    A simple tokenizer that counts the token by the splitted space
    Then a rough estimate of the token count is returned. (times 1.5)
    Args:
        text (str): The input text to tokenize.
    """
    return len(text.split(" ")) * 1.5


def calculate_token_throughput(token_count: int, duration: float) -> float:
    """
    Calculate the token throughput.

    Args:
        token_count (int): The number of tokens processed.
        duration (float): The time taken to process the tokens in seconds.

    Returns:
        float: The token throughput in tokens per second.
    """
    if duration <= 0:
        return 0.0
    return token_count / duration


def _json_default_serializer(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def _persist_metrics(metrics: Dict[str, Any], metrics_path: Union[str, Path]) -> None:
    destination = Path(metrics_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    existing: List[Dict[str, Any]] = []
    if destination.exists():
        try:
            with destination.open("r", encoding="utf-8") as handle:
                content = handle.read().strip()
            if content:
                loaded = json.loads(content)
                if isinstance(loaded, list):
                    existing = loaded
                else:
                    existing = [loaded]
        except json.JSONDecodeError:
            eval_logger.warning(f"Existing metrics file {destination} is not valid JSON. Overwriting.")
        except Exception as exc:
            eval_logger.warning(f"Could not read metrics file at {destination}: {exc}")

    existing.append(metrics)

    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(existing, handle, indent=2, ensure_ascii=False, default=_json_default_serializer)
        handle.write("\n")
    tmp_path.replace(destination)


def log_metrics(
    e2e_latency: float,
    total_tokens: int,
    avg_speed: float,
    additional_metrics: Optional[Dict[str, Any]] = None,
    metrics_path: Optional[Union[str, Path]] = None,
):
    """
    Log the metrics in a structured format and optionally persist them to disk.

    Args:
        e2e_latency (float): The end-to-end latency in seconds.
        total_tokens (int): The total number of tokens processed.
        avg_speed (float): The average speed in tokens per second.
        additional_metrics (Dict[str, Any], optional): Additional metrics to log.
        metrics_path (Union[str, Path], optional): Path to a JSON file where metrics should be appended.
            If not provided, the `LMMS_EVAL_METRICS_PATH` environment variable will be used when available.
    """
    required_stats = f"Metric summary - Total time: {e2e_latency:.3f}s, Total tokens: {total_tokens}, Avg speed: {avg_speed:.1f} tokens/s"
    if additional_metrics:
        required_stats += ", Additional metrics: "
        required_stats += ", ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in additional_metrics.items())
    eval_logger.info(required_stats)

    metrics_payload: Dict[str, Any] = {
        "e2e_latency": float(e2e_latency),
        "total_tokens": float(total_tokens),
        "avg_speed": float(avg_speed),
        "logged_at": time.time(),
    }
    if additional_metrics:
        metrics_payload["additional_metrics"] = additional_metrics

    destination = metrics_path or os.getenv("LMMS_EVAL_METRICS_PATH")
    if destination:
        try:
            _persist_metrics(metrics_payload, destination)
        except Exception as exc:
            eval_logger.warning(f"Failed to persist metrics to {destination}: {exc}")


class GenMetrics:
    """
    A class to manage the generation of metrics for model evaluation.
    """

    def __init__(self, tokenize_fn: Callable = space_tokenizer, metrics_path: Optional[Union[str, Path]] = None):
        self.tokenize_fn = tokenize_fn
        self.metrics_path = Path(metrics_path) if metrics_path else None

    def __enter__(self):
        """
        Initialize the context manager.
        """
        self.metrics = {}
        self.start_time = time.perf_counter()
        return self

    def stop_timer(self):
        self.end_time = time.perf_counter()

    def log_metric(self, content: List[Any], additional_metrics: Dict[str, Any] = None):
        num_tokens = sum(self.tokenize_fn(item) for item in content)
        duration = self.end_time - self.start_time
        throughput = calculate_token_throughput(num_tokens, duration)
        base_metrics = {
            "num_tokens": num_tokens,
            "duration": duration,
            "throughput": throughput,
        }
        self.metrics = base_metrics.copy()
        if additional_metrics:
            self.metrics.update(additional_metrics)

        supplementary_metrics = {k: v for k, v in self.metrics.items() if k not in {"num_tokens", "duration", "throughput"}}
        log_metrics(
            e2e_latency=duration,
            total_tokens=num_tokens,
            avg_speed=throughput,
            additional_metrics=supplementary_metrics or None,
            metrics_path=self.metrics_path,
        )

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Finalize the context manager and return the collected metrics.
        """
        self.end_time = time.perf_counter()
        self.metrics["duration"] = self.end_time - self.start_time
        return self.metrics
