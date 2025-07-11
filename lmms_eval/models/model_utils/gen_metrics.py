import time
from typing import Any, Callable, Dict, List

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


def log_metrics(e2e_latency: float, total_tokens: int, avg_speed: float, additional_metrics: Dict[str, Any] = None):
    """
    Log the metrics in a structured format.

    Args:
        e2e_latency (float): The end-to-end latency in seconds.
        total_tokens (int): The total number of tokens processed.
        avg_speed (float): The average speed in tokens per second.
        additional_metrics (Dict[str, Any]): Additional metrics to log.
    """
    required_stats = f"Metric summary - Total time: {e2e_latency:.3f}s, Total tokens: {total_tokens}, Avg speed: {avg_speed:.1f} tokens/s"
    if additional_metrics is not None:
        required_stats += ", Additional metrics: "
        required_stats += ", ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in additional_metrics.items())
    eval_logger.info(required_stats)


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

    def log_metric(self, content: List[Any], additional_metrics: Dict[str, Any] = None):
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

        log_metrics(self.metrics)

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Finalize the context manager and return the collected metrics.
        """
        self.end_time = time.perf_counter()
        self.metrics["duration"] = self.end_time - self.start_time
        return self.metrics
