"""Layout configuration for lmms-eval terminal UI."""

from rich.layout import Layout
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text


def create_layout():
    """Create the main layout for the lmms-eval dashboard."""
    layout = Layout()

    # Split the layout into main sections
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="progress", size=6),
        Layout(name="metrics", ratio=1),
        Layout(name="tasks", ratio=1),
        Layout(name="logs", size=10),
    )

    # Split progress into overall and current task progress
    layout["progress"].split_column(
        Layout(name="overall_progress"),
        Layout(name="task_progress"),
        Layout(name="request_progress"),
    )

    # Split metrics into evaluation metrics
    layout["metrics"].split_row(
        Layout(name="eval_metrics"),
        Layout(name="performance_metrics"),
    )

    # Task list and current task details
    layout["tasks"].split_row(
        Layout(name="task_list", ratio=1),
        Layout(name="current_task", ratio=2),
    )

    return layout


def create_metric_panel(title, metrics_data):
    """Create a panel displaying metrics."""
    table = Table.grid(expand=True)
    table.add_column(justify="left")

    if metrics_data:
        rows = []
        for key, value in metrics_data.items():
            if isinstance(value, (int, float)):
                rows.append(f"[yellow]{key}:[/yellow] {value:.4f}")
            else:
                rows.append(f"[yellow]{key}:[/yellow] {value}")

        table.add_row(Text.from_markup("\n".join(rows), justify="left"))
    else:
        table.add_row(Text("No metrics available yet", justify="left"))

    return Panel(table, title=title, border_style="blue")


def create_progress_bars():
    """Create progress bars for the dashboard."""
    overall_progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        expand=True,
    )

    task_progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TextColumn("{task.completed}/{task.total}"),
        expand=True,
    )

    request_progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} requests"),
        expand=True,
    )

    # Add initial tasks
    overall_task = overall_progress.add_task("Overall Evaluation", total=100)

    return overall_progress, task_progress, request_progress, overall_task


def update_progress(layout, overall_progress, task_progress, request_progress):
    """Update progress panels in the layout."""
    layout["overall_progress"].update(Panel(overall_progress, title="Overall Progress", border_style="magenta"))
    layout["task_progress"].update(Panel(task_progress, title="Current Task Progress", border_style="cyan"))
    layout["request_progress"].update(Panel(request_progress, title="Request Processing", border_style="green"))


def create_task_list_panel(task_names, current_task_idx=None):
    """Create a panel showing the list of tasks."""
    table = Table.grid(expand=True)
    table.add_column(justify="left", width=3)
    table.add_column(justify="left")

    for idx, task_name in enumerate(task_names):
        if idx == current_task_idx:
            table.add_row("▶", f"[bold cyan]{task_name}[/bold cyan]")
        else:
            table.add_row(" ", task_name)

    return Panel(table, title="Task List", border_style="yellow")


def create_header_panel(model_name, model_args):
    """Create header panel with model information."""
    header_text = f"[bold blue]LMMS-Eval Dashboard[/bold blue]\n"
    header_text += f"Model: [green]{model_name}[/green]\n"
    if model_args:
        header_text += f"Args: [dim]{model_args}[/dim]"

    return Panel(
        Text.from_markup(header_text, justify="center"),
        border_style="bold blue",
    )
