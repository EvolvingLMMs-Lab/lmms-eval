"""Plotting utilities for lmms-eval terminal UI."""

import math
from typing import List, Optional

import numpy as np
from loguru import logger as eval_logger
from rich.text import Text


def create_horizontal_bar_chart(values: List[float], labels: Optional[List[str]] = None, width: int = 40, max_value: Optional[float] = None) -> Text:
    """Create a horizontal bar chart for metrics display."""
    if not values:
        return Text("No data available")

    if max_value is None:
        max_value = max(values) if values else 1.0

    if max_value == 0:
        max_value = 1.0

    chart = Text()

    # If labels not provided, create default labels
    if labels is None:
        labels = [f"Metric {i+1}" for i in range(len(values))]

    # Determine the width of the longest label
    max_label_width = max(len(label) for label in labels) if labels else 10

    for label, value in zip(labels, values):
        bar_length = int((value / max_value) * width)
        bar = "█" * bar_length

        # Color based on value
        if value < max_value * 0.33:
            color = "red"
        elif value < max_value * 0.66:
            color = "yellow"
        else:
            color = "green"

        chart.append(f"{label:<{max_label_width}} | ", style="bold")
        chart.append(bar, style=color)
        chart.append(f" {value:.4f}\n")

    return chart


def create_accuracy_histogram(accuracies: List[float], bin_width: float = 0.1) -> Text:
    """Create histogram for accuracy distribution."""
    if not accuracies:
        eval_logger.warning("No accuracy data for histogram.")
        return Text("No data available")

    # Create bins from 0 to 1
    bins = np.arange(0, 1.1, bin_width)
    hist, bin_edges = np.histogram(accuracies, bins=bins)

    # Create the histogram chart
    chart = Text()
    chart.append("Accuracy Distribution\n", style="bold")

    for i, count in enumerate(hist):
        if count > 0:
            bin_start = bin_edges[i]
            bin_end = bin_edges[i + 1]
            bar = "█" * int(count)
            chart.append(f"{bin_start:.1f}-{bin_end:.1f} | {bar} {count}\n")

    return chart


def create_task_metrics_chart(task_metrics: dict, width: int = 40) -> Text:
    """Create a chart showing metrics for different tasks."""
    if not task_metrics:
        return Text("No task metrics available")

    chart = Text()

    # Extract task names and scores
    tasks = list(task_metrics.keys())
    scores = []

    for task, metrics in task_metrics.items():
        # Try to find the main metric (could be accuracy, score, etc.)
        if isinstance(metrics, dict):
            score = metrics.get("accuracy", metrics.get("score", 0.0))
        else:
            score = float(metrics)
        scores.append(score)

    # Create the bar chart
    max_score = max(scores) if scores else 1.0
    max_task_width = max(len(task) for task in tasks) if tasks else 10

    for task, score in zip(tasks, scores):
        bar_length = int((score / max_score) * width)
        bar = "█" * bar_length

        # Color coding
        if score < 0.5:
            color = "red"
        elif score < 0.8:
            color = "yellow"
        else:
            color = "green"

        chart.append(f"{task:<{max_task_width}} | ", style="bold")
        chart.append(bar, style=color)
        chart.append(f" {score:.3f}\n")

    return chart


def create_mini_sparkline(values: List[float], width: int = 20, height: int = 5) -> Text:
    """Create a mini sparkline chart for showing trends."""
    if not values or len(values) < 2:
        return Text("—" * width)

    # Normalize values to height
    min_val = min(values)
    max_val = max(values)

    if min_val == max_val:
        # All values are the same
        return Text("—" * width)

    # Sample values if too many
    if len(values) > width:
        indices = np.linspace(0, len(values) - 1, width, dtype=int)
        sampled_values = [values[i] for i in indices]
    else:
        sampled_values = values

    # Create sparkline using Unicode block characters
    blocks = " ▁▂▃▄▅▆▇█"

    sparkline = ""
    for val in sampled_values:
        normalized = (val - min_val) / (max_val - min_val)
        block_idx = int(normalized * (len(blocks) - 1))
        sparkline += blocks[block_idx]

    return Text(sparkline)
