"""Dashboard implementation for lmms-eval terminal UI."""

import os
import time
from typing import Any, Dict, List, Optional, Union

from loguru import logger as eval_logger
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .layout import (
    create_header_panel,
    create_layout,
    create_metric_panel,
    create_progress_bars,
    create_task_list_panel,
    update_progress,
)
from .metrics import MetricsCollector, format_metrics_for_display
from .plots import (
    create_accuracy_histogram,
    create_horizontal_bar_chart,
    create_task_metrics_chart,
)


class MinimalDashboard:
    """Minimal dashboard implementation for when UI is disabled."""

    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self._live = type(
            "MinimalDashboardLive",
            (),
            {
                "__enter__": lambda x: None,
                "__exit__": lambda x, *args: None,
            },
        )()

    @property
    def live(self):
        return self._live

    def start_evaluation(self, model_name: str, model_args: str, tasks: List[str]):
        pass

    def start_task(self, task_name: str, total_samples: int):
        self.metrics_collector.start_task(task_name, total_samples)

    def update_task_progress(self, task_name: str, processed: int):
        self.metrics_collector.update_task_progress(task_name, processed)

    def update_request_processing(self, request_type: str, completed: int, total: int):
        pass

    def add_task_metrics(self, task_name: str, metrics: Dict[str, Any]):
        self.metrics_collector.update_task_metrics(task_name, metrics)

    def end_task(self, task_name: str):
        self.metrics_collector.end_task(task_name)

    def show_final_results(self, results: Dict[str, Any]):
        pass


class RichDashboard:
    """Rich terminal UI dashboard for lmms-eval."""

    def __init__(self):
        self.console = Console()
        self.layout = create_layout()
        self.metrics_collector = MetricsCollector()

        # Progress bars
        (
            self.overall_progress,
            self.task_progress,
            self.request_progress,
            self.overall_task_id,
        ) = create_progress_bars()

        # Task tracking
        self.tasks: List[str] = []
        self.current_task_idx: Optional[int] = None
        self.task_progress_ids: Dict[str, int] = {}
        self.request_progress_id: Optional[int] = None

        # Model information
        self.model_name: str = ""
        self.model_args: str = ""

        # Live display
        self.live: Live = Live(
            self.layout,
            console=self.console,
            refresh_per_second=4,
            screen=True,
        )

        # Metrics tracking
        self.task_accuracies: List[float] = []
        self.log_messages: List[str] = []
        self.max_log_messages = 20

    def start_evaluation(self, model_name: str, model_args: str, tasks: List[str]):
        """Initialize the dashboard for evaluation."""
        self.model_name = model_name
        self.model_args = model_args
        self.tasks = tasks

        # Update header
        self.layout["header"].update(create_header_panel(model_name, model_args))

        # Update task list
        self.layout["task_list"].update(create_task_list_panel(tasks))

        # Initialize empty panels
        self._initialize_panels()

        # Add log message
        self.add_log(f"Starting evaluation with {len(tasks)} tasks")

    def _initialize_panels(self):
        """Initialize all panels with empty content."""
        self.layout["eval_metrics"].update(Panel(Text("Waiting for metrics..."), title="Evaluation Metrics", border_style="blue"))
        self.layout["performance_metrics"].update(Panel(Text("Waiting for metrics..."), title="Performance Metrics", border_style="green"))
        self.layout["current_task"].update(Panel(Text("No task running"), title="Current Task", border_style="yellow"))
        self.layout["logs"].update(Panel(Text(""), title="Logs", border_style="dim"))
        update_progress(self.layout, self.overall_progress, self.task_progress, self.request_progress)

    def start_task(self, task_name: str, total_samples: int):
        """Start a new task."""
        self.metrics_collector.start_task(task_name, total_samples)

        # Update current task index
        if task_name in self.tasks:
            self.current_task_idx = self.tasks.index(task_name)
            self.layout["task_list"].update(create_task_list_panel(self.tasks, self.current_task_idx))

        # Add task progress bar
        task_id = self.task_progress.add_task(task_name, total=total_samples)
        self.task_progress_ids[task_name] = task_id

        # Update current task panel
        self._update_current_task_panel(task_name, 0, total_samples)

        # Update overall progress
        completed_tasks = sum(1 for t in self.task_progress_ids if t not in [self.metrics_collector.current_task])
        self.overall_progress.update(self.overall_task_id, completed=(completed_tasks / len(self.tasks) * 100) if self.tasks else 0)

        # Add log
        self.add_log(f"Started task: {task_name} ({total_samples} samples)")

        update_progress(self.layout, self.overall_progress, self.task_progress, self.request_progress)

    def update_task_progress(self, task_name: str, processed: int):
        """Update progress for a task."""
        self.metrics_collector.update_task_progress(task_name, processed)

        if task_name in self.task_progress_ids:
            self.task_progress.update(self.task_progress_ids[task_name], completed=processed)

        # Update current task panel
        task_metrics = self.metrics_collector.task_metrics.get(task_name)
        if task_metrics:
            self._update_current_task_panel(task_name, processed, task_metrics.total_samples)

        # Update metrics panels
        self._update_metrics_panels()

        update_progress(self.layout, self.overall_progress, self.task_progress, self.request_progress)

    def update_request_processing(self, request_type: str, completed: int, total: int):
        """Update request processing progress."""
        if self.request_progress_id is None:
            self.request_progress_id = self.request_progress.add_task(f"Processing {request_type} requests", total=total)
        else:
            self.request_progress.update(self.request_progress_id, description=f"Processing {request_type} requests", total=total, completed=completed)

        update_progress(self.layout, self.overall_progress, self.task_progress, self.request_progress)

    def add_task_metrics(self, task_name: str, metrics: Dict[str, Any]):
        """Add metrics for a task."""
        self.metrics_collector.update_task_metrics(task_name, metrics)

        # Track accuracy if available
        if "accuracy" in metrics:
            accuracy = float(metrics["accuracy"])
            self.task_accuracies.append(accuracy)

        self._update_metrics_panels()

    def end_task(self, task_name: str):
        """Mark a task as completed."""
        self.metrics_collector.end_task(task_name)

        # Update task progress to 100%
        if task_name in self.task_progress_ids:
            task_metrics = self.metrics_collector.task_metrics.get(task_name)
            if task_metrics:
                self.task_progress.update(self.task_progress_ids[task_name], completed=task_metrics.total_samples)

        # Update overall progress
        completed_tasks = len([t for t in self.metrics_collector.task_metrics.values() if t.end_time is not None])
        self.overall_progress.update(self.overall_task_id, completed=(completed_tasks / len(self.tasks) * 100) if self.tasks else 0)

        # Clear request progress
        if self.request_progress_id is not None:
            self.request_progress.remove_task(self.request_progress_id)
            self.request_progress_id = None

        # Add log
        self.add_log(f"Completed task: {task_name}")

        update_progress(self.layout, self.overall_progress, self.task_progress, self.request_progress)

    def _update_current_task_panel(self, task_name: str, processed: int, total: int):
        """Update the current task panel."""
        task_metrics = self.metrics_collector.task_metrics.get(task_name)

        if task_metrics:
            table = Table.grid(expand=True)
            table.add_column(justify="left")

            info = [
                f"[bold]Task:[/bold] {task_name}",
                f"[bold]Progress:[/bold] {processed}/{total} ({task_metrics.progress_percentage:.1f}%)",
                f"[bold]Speed:[/bold] {task_metrics.samples_per_second:.2f} samples/s",
                f"[bold]Elapsed:[/bold] {task_metrics.elapsed_time:.1f}s",
            ]

            # Add task-specific metrics
            if task_metrics.metrics:
                info.append("\n[bold]Metrics:[/bold]")
                for key, value in task_metrics.metrics.items():
                    if isinstance(value, float):
                        info.append(f"  {key}: {value:.4f}")

            table.add_row(Text.from_markup("\n".join(info)))

            self.layout["current_task"].update(Panel(table, title="Current Task", border_style="yellow"))

    def _update_metrics_panels(self):
        """Update metrics panels."""
        current_metrics = self.metrics_collector.get_current_metrics()

        # Evaluation metrics
        eval_metrics = {
            "Tasks Completed": f"{current_metrics.get('tasks_completed', 0)}/{current_metrics.get('total_tasks', 0)}",
            "Total Samples": current_metrics.get("total_samples_processed", 0),
        }

        # Add accuracy if available
        if self.task_accuracies:
            eval_metrics["Avg Accuracy"] = f"{sum(self.task_accuracies) / len(self.task_accuracies):.3f}"

        self.layout["eval_metrics"].update(create_metric_panel("Evaluation Metrics", eval_metrics))

        # Performance metrics
        perf_metrics = {
            "Elapsed Time": f"{current_metrics.get('overall_elapsed', 0):.1f}s",
        }

        if "current_task_speed" in current_metrics:
            perf_metrics["Current Speed"] = f"{current_metrics['current_task_speed']:.2f} samples/s"

        self.layout["performance_metrics"].update(create_metric_panel("Performance Metrics", perf_metrics))

    def add_log(self, message: str):
        """Add a log message."""
        timestamp = time.strftime("%H:%M:%S")
        self.log_messages.append(f"[dim]{timestamp}[/dim] {message}")

        # Keep only recent messages
        if len(self.log_messages) > self.max_log_messages:
            self.log_messages = self.log_messages[-self.max_log_messages :]

        # Update log panel
        log_text = "\n".join(self.log_messages)
        self.layout["logs"].update(Panel(Text.from_markup(log_text), title="Logs", border_style="dim"))

    def show_final_results(self, results: Dict[str, Any]):
        """Display final evaluation results."""
        # Create results visualization
        task_results = self.metrics_collector.get_task_results()

        if task_results:
            # Create task metrics chart
            task_scores = {}
            for task_name, metrics in task_results.items():
                # Try to find the main score
                score = metrics.get("accuracy", metrics.get("score", 0.0))
                task_scores[task_name] = score

            chart = create_task_metrics_chart(task_scores)
            self.layout["current_task"].update(Panel(chart, title="Task Results", border_style="green"))

        # Update final metrics
        self._update_metrics_panels()

        # Add completion log
        self.add_log("Evaluation completed!")

        # Add final message to inform user that evaluation is complete
        self.add_log("Press Ctrl+C to exit")

        # Keep the display alive indefinitely until user interrupts
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


Dashboard = Union[RichDashboard, MinimalDashboard]


def create_dashboard() -> Dashboard:
    """Factory function to create appropriate dashboard based on ENABLE_UI env var."""
    enable_ui = os.getenv("ENABLE_UI", "true").lower() == "true"
    return RichDashboard() if enable_ui else MinimalDashboard()
