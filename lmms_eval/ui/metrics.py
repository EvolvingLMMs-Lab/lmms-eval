"""Metrics adapters and collectors for lmms-eval UI."""

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""

    task_name: str
    total_samples: int
    processed_samples: int
    accuracy: Optional[float] = None
    metrics: Dict[str, float] = None
    start_time: float = None
    end_time: float = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.start_time is None:
            self.start_time = time.time()

    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    @property
    def samples_per_second(self) -> float:
        """Calculate processing speed."""
        elapsed = self.elapsed_time
        if elapsed > 0:
            return self.processed_samples / elapsed
        return 0.0

    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage."""
        if self.total_samples > 0:
            return (self.processed_samples / self.total_samples) * 100
        return 0.0


class MetricsCollector:
    """Collects and manages metrics during evaluation."""

    def __init__(self):
        self.task_metrics: Dict[str, EvaluationMetrics] = {}
        self.request_metrics: Dict[str, List[float]] = defaultdict(list)
        self.overall_start_time = time.time()
        self.current_task: Optional[str] = None

    def start_task(self, task_name: str, total_samples: int):
        """Start tracking a new task."""
        self.current_task = task_name
        self.task_metrics[task_name] = EvaluationMetrics(task_name=task_name, total_samples=total_samples, processed_samples=0)

    def update_task_progress(self, task_name: str, processed: int):
        """Update task progress."""
        if task_name in self.task_metrics:
            self.task_metrics[task_name].processed_samples = processed

    def add_request_metric(self, metric_name: str, value: float):
        """Add a request-level metric."""
        self.request_metrics[metric_name].append(value)

    def update_task_metrics(self, task_name: str, metrics: Dict[str, Any]):
        """Update metrics for a task."""
        if task_name in self.task_metrics:
            # Convert metrics to float where possible
            processed_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    processed_metrics[key] = float(value)
                elif isinstance(value, list) and value:
                    # If it's a list, take the mean
                    try:
                        processed_metrics[key] = sum(value) / len(value)
                    except:
                        pass

            self.task_metrics[task_name].metrics.update(processed_metrics)

            # Update accuracy if present
            if "accuracy" in processed_metrics:
                self.task_metrics[task_name].accuracy = processed_metrics["accuracy"]

    def end_task(self, task_name: str):
        """Mark a task as completed."""
        if task_name in self.task_metrics:
            self.task_metrics[task_name].end_time = time.time()

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics for UI display."""
        metrics = {
            "overall_elapsed": time.time() - self.overall_start_time,
            "tasks_completed": sum(1 for t in self.task_metrics.values() if t.end_time is not None),
            "total_tasks": len(self.task_metrics),
            "current_task": self.current_task,
        }

        # Add current task metrics
        if self.current_task and self.current_task in self.task_metrics:
            current = self.task_metrics[self.current_task]
            metrics["current_task_progress"] = current.progress_percentage
            metrics["current_task_speed"] = current.samples_per_second

        # Calculate overall metrics
        total_samples = sum(t.processed_samples for t in self.task_metrics.values())
        metrics["total_samples_processed"] = total_samples

        # Add request metrics statistics
        for metric_name, values in self.request_metrics.items():
            if values:
                metrics[f"{metric_name}_mean"] = sum(values) / len(values)
                metrics[f"{metric_name}_min"] = min(values)
                metrics[f"{metric_name}_max"] = max(values)

        return metrics

    def get_task_results(self) -> Dict[str, Dict[str, float]]:
        """Get final results for all tasks."""
        results = {}
        for task_name, task_metric in self.task_metrics.items():
            results[task_name] = {
                "processed_samples": task_metric.processed_samples,
                "total_samples": task_metric.total_samples,
                "elapsed_time": task_metric.elapsed_time,
                "samples_per_second": task_metric.samples_per_second,
                **task_metric.metrics,
            }
        return results


def format_metrics_for_display(metrics: Dict[str, Any]) -> Dict[str, str]:
    """Format metrics for UI display."""
    formatted = {}

    for key, value in metrics.items():
        if isinstance(value, float):
            # Format based on metric type
            if "time" in key or "elapsed" in key:
                # Format as time
                formatted[key] = f"{value:.2f}s"
            elif "percentage" in key or "accuracy" in key:
                # Format as percentage
                formatted[key] = f"{value:.1f}%"
            elif "speed" in key or "per_second" in key:
                # Format as rate
                formatted[key] = f"{value:.2f}/s"
            else:
                # General float formatting
                formatted[key] = f"{value:.4f}"
        elif isinstance(value, int):
            formatted[key] = str(value)
        else:
            formatted[key] = str(value)

    return formatted
