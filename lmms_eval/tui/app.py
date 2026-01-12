"""
LMMs-Eval TUI Application
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from importlib.metadata import version as pkg_version
from typing import ClassVar

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Grid, Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    OptionList,
    Pretty,
    Rule,
    Select,
    Static,
    Switch,
    TabbedContent,
    TabPane,
)
from textual.widgets.option_list import Option

try:
    from textual_plotext import PlotextPlot

    HAS_PLOTEXT = True
except ImportError:
    HAS_PLOTEXT = False

try:
    from rich_pixels import Pixels
    from PIL import Image

    HAS_RICH_PIXELS = True
except ImportError:
    HAS_RICH_PIXELS = False

LOGO_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "assets", "logo.png")


def get_version() -> str:
    """Get lmms-eval version from package metadata."""
    try:
        return pkg_version("lmms_eval")
    except Exception:
        return "0.5.0"


# URLs for lmms-eval
LMMS_EVAL_REPO = "https://github.com/EvolvingLMMs-Lab/lmms-eval"
LMMS_EVAL_ISSUES = f"{LMMS_EVAL_REPO}/issues"
LMMS_EVAL_PRS = f"{LMMS_EVAL_REPO}/pulls"


@dataclass
class EvalConfig:
    model: str = "openai_compatible"
    model_args: str = ""
    tasks: list[str] = field(default_factory=list)
    batch_size: int = 1
    limit: int | None = None
    output_path: str = "./logs/"
    log_samples: bool = True
    verbosity: str = "INFO"
    device: str | None = None
    num_fewshot: int | None = None


POPULAR_MODELS = [
    ("openai_compatible", "OpenAI Compatible API"),
    ("qwen2_5_vl", "Qwen2.5-VL"),
    ("qwen2_5_vl_chat", "Qwen2.5-VL Chat"),
    ("llava_onevision", "LLaVA-OneVision"),
    ("llava", "LLaVA"),
    ("internvl2", "InternVL2"),
    ("claude", "Claude API"),
    ("gemini_api", "Gemini API"),
]

POPULAR_TASKS = [
    ("mme", "MME - Multimodal Evaluation"),
    ("mmmu_val", "MMMU Validation"),
    ("scienceqa_img", "ScienceQA (Image)"),
    ("mathvista_testmini", "MathVista TestMini"),
    ("ai2d", "AI2D"),
    ("chartqa", "ChartQA"),
    ("docvqa_val", "DocVQA Validation"),
    ("textvqa_val", "TextVQA Validation"),
    ("ocrbench", "OCRBench"),
    ("realworldqa", "RealWorldQA"),
]


# Big ASCII art logo for welcome screen
LOGO = r"""
[#5dadec]██╗      ███╗   ███╗███╗   ███╗ ███████╗[/#5dadec]
[#5dadec]██║      ████╗ ████║████╗ ████║██╔════╝[/#5dadec] 
[#5dadec]██║      ██╔████╔██║██╔████╔██║███████╗[/#5dadec] 
[#5dadec]██║      ██║╚██╔╝██║██║╚██╔╝██║╚════██║[/#5dadec] 
[#5dadec]███████╗ ██║ ╚═╝ ██║██║ ╚═╝ ██║███████║[/#5dadec]
[#4a8bc2]╚══════╝ ╚═╝     ╚═╝╚═╝     ╚═╝╚══════╝[/#4a8bc2]

[#5dadec]███████╗██╗   ██╗  █████╗ ██╗[/#5dadec]           
[#5dadec]██╔════╝██║   ██║ ██╔══██╗██║[/#5dadec]           
[#5dadec]█████╗  ██║   ██║ ███████║██║[/#5dadec]           
[#5dadec]██╔══╝  ╚██╗ ██╔╝ ██╔══██║██║[/#5dadec]           
[#5dadec]███████╗ ╚████╔╝  ██║  ██║███████╗[/#5dadec]      
[#4a8bc2]╚══════╝ ╚═══╝    ╚═╝  ╚═╝╚══════╝[/#4a8bc2]      
"""


class LogoWidget(Static):
    def __init__(self, width: int = 30, **kwargs) -> None:
        super().__init__(**kwargs)
        self.logo_width = width

    def render(self):
        if HAS_RICH_PIXELS:
            try:
                if os.path.exists(LOGO_IMAGE_PATH):
                    img = Image.open(LOGO_IMAGE_PATH)
                    if img.mode == "RGBA":
                        bg = Image.new("RGB", img.size, (30, 30, 30))
                        bg.paste(img, mask=img.split()[3])
                        img = bg
                    aspect = img.height / img.width
                    new_width = self.logo_width
                    new_height = int(new_width * aspect * 2)
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    return Pixels.from_image(img)
            except Exception:
                pass
        return "[#5dadec]◆ LMMs[/#5dadec]\n[#5dadec]  Eval[/#5dadec]"


class WelcomeScreen(Screen):
    BINDINGS = [
        Binding("enter", "start", "Start Configuration"),
        Binding("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        version = get_version()
        yield Container(
            Static(
                "[yellow]✱[/yellow] Welcome to [bold #5dadec]LMMs-Eval[/bold #5dadec] "
                "interactive evaluator!",
                id="welcome-msg",
            ),
            Static(LOGO, id="logo"),
            Static(
                f"[bold #5dadec]v{version}[/bold #5dadec]",
                id="version-info",
            ),
            Static("[dim]────────────────────────────────────────[/dim]"),
            Static(f"[dim]🏠 Homepage:[/dim] [#5dadec]{LMMS_EVAL_REPO}[/#5dadec]"),
            Static(f"[dim]🐛 Bug Report:[/dim] [#5dadec]{LMMS_EVAL_ISSUES}[/#5dadec]"),
            Static(f"[dim]🔀 Pull Requests:[/dim] [#5dadec]{LMMS_EVAL_PRS}[/#5dadec]"),
            Static("[dim]────────────────────────────────────────[/dim]"),
            Static(
                "[dim]🎉 Press [bold white]Enter[/bold white] to continue[/dim]",
                id="continue-msg",
            ),
            id="welcome-container",
        )

    def action_start(self) -> None:
        self.app.push_screen(ConfigScreen())

    def action_quit(self) -> None:
        self.app.exit()

    def on_key(self, event) -> None:
        if event.key == "enter":
            self.action_start()
        elif event.key == "q":
            self.action_quit()


class ConfigScreen(Screen):
    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("ctrl+r", "run", "Run Evaluation"),
    ]

    def compose(self) -> ComposeResult:
        version = get_version()
        yield Header()
        with Horizontal(id="welcome-header"):
            with Horizontal(id="header-left"):
                yield LogoWidget(id="header-logo")
                with Vertical(id="header-info"):
                    yield Static(
                        f"[bold #5dadec]LMMs-Eval[/bold #5dadec] [dim]v{version}[/dim]"
                    )
                    yield Static(f"[dim]Python {sys.version.split()[0]}[/dim]")
                    yield Static(f"[dim]{LMMS_EVAL_REPO}[/dim]")
            with Vertical(id="header-right"):
                yield Static("[bold]Command Preview[/bold]", id="preview-title")
                yield Static("", id="command-preview")
        with Container(id="config-container"):
            with TabbedContent():
                with TabPane("Model", id="model-tab"):
                    with VerticalScroll():
                        yield Static("[bold]Select Model[/]", classes="section-title")
                        yield Select(
                            [(name, key) for key, name in POPULAR_MODELS],
                            prompt="Choose a model",
                            id="model-select",
                        )
                        yield Rule()
                        yield Static(
                            "[bold]Model Arguments[/]", classes="section-title"
                        )
                        yield Static(
                            "[dim]e.g., model_version=gpt-4o,pretrained=path/to/model[/]"
                        )
                        yield Input(
                            placeholder="model_args (comma-separated key=value pairs)",
                            id="model-args-input",
                        )
                        yield Rule()
                        yield Static(
                            "[bold]API Configuration (for API models)[/]",
                            classes="section-title",
                        )
                        yield Static(
                            "[dim]Set OPENAI_API_KEY and OPENAI_API_BASE in environment[/]"
                        )
                        yield Input(
                            placeholder="API Base URL (optional)", id="api-base-input"
                        )
                with TabPane("Tasks", id="tasks-tab"):
                    with VerticalScroll():
                        yield Static("[bold]Search Tasks[/]", classes="section-title")
                        yield Input(
                            placeholder="Type to filter tasks...", id="task-search"
                        )
                        yield Rule()
                        yield Static("[bold]Popular Tasks[/]", classes="section-title")
                        yield OptionList(
                            *[
                                Option(f"{name} ({key})", id=key)
                                for key, name in POPULAR_TASKS
                            ],
                            id="task-list",
                        )
                        yield Rule()
                        yield Static("[bold]Selected Tasks[/]", classes="section-title")
                        yield Static("None selected", id="selected-tasks-display")
                with TabPane("Settings", id="settings-tab"):
                    with VerticalScroll():
                        yield Static(
                            "[bold]Evaluation Settings[/]", classes="section-title"
                        )
                        yield Horizontal(
                            Static("Batch Size:", classes="setting-label"),
                            Input(
                                value="1",
                                id="batch-size-input",
                                classes="setting-input",
                            ),
                            classes="setting-row",
                        )
                        yield Horizontal(
                            Static("Limit (samples):", classes="setting-label"),
                            Input(
                                placeholder="None (all samples)",
                                id="limit-input",
                                classes="setting-input",
                            ),
                            classes="setting-row",
                        )
                        yield Horizontal(
                            Static("Output Path:", classes="setting-label"),
                            Input(
                                value="./logs/",
                                id="output-path-input",
                                classes="setting-input",
                            ),
                            classes="setting-row",
                        )
                        yield Horizontal(
                            Static("Device:", classes="setting-label"),
                            Input(
                                placeholder="auto (cuda:0, cpu, etc.)",
                                id="device-input",
                                classes="setting-input",
                            ),
                            classes="setting-row",
                        )
                        yield Rule()
                        yield Static("[bold]Options[/]", classes="section-title")
                        yield Horizontal(
                            Static("Log Samples:", classes="setting-label"),
                            Switch(value=True, id="log-samples-switch"),
                            classes="setting-row",
                        )
                        yield Horizontal(
                            Static("Verbosity:", classes="setting-label"),
                            Select(
                                [
                                    ("DEBUG", "DEBUG"),
                                    ("INFO", "INFO"),
                                    ("WARNING", "WARNING"),
                                    ("ERROR", "ERROR"),
                                ],
                                value="INFO",
                                id="verbosity-select",
                            ),
                            classes="setting-row",
                        )
                with TabPane("Preview", id="preview-tab"):
                    with Container(id="preview-container"):
                        yield Static(
                            "[bold]Command Preview[/]", classes="section-title"
                        )
                        with VerticalScroll(id="command-scroll"):
                            yield Static(
                                "", id="command-preview", classes="command-preview"
                            )
                        with Horizontal(id="preview-buttons"):
                            yield Button(
                                "▶ Run Evaluation", variant="success", id="run-btn"
                            )
                            yield Button(
                                "📋 Copy Command", variant="primary", id="copy-btn"
                            )
        yield Footer()

    @on(Select.Changed, "#model-select")
    def on_model_changed(self, event: Select.Changed) -> None:
        self.app.config.model = str(event.value) if event.value else "openai_compatible"
        self._update_preview()

    @on(Input.Changed, "#model-args-input")
    def on_model_args_changed(self, event: Input.Changed) -> None:
        self.app.config.model_args = event.value
        self._update_preview()

    @on(OptionList.OptionSelected, "#task-list")
    def on_task_selected(self, event: OptionList.OptionSelected) -> None:
        task_id = str(event.option.id)
        if task_id not in self.app.config.tasks:
            self.app.config.tasks.append(task_id)
        else:
            self.app.config.tasks.remove(task_id)
        self._update_selected_tasks()
        self._update_preview()

    @on(Input.Changed, "#batch-size-input")
    def on_batch_size_changed(self, event: Input.Changed) -> None:
        try:
            self.app.config.batch_size = int(event.value) if event.value else 1
        except ValueError:
            pass
        self._update_preview()

    @on(Input.Changed, "#limit-input")
    def on_limit_changed(self, event: Input.Changed) -> None:
        try:
            self.app.config.limit = int(event.value) if event.value else None
        except ValueError:
            self.app.config.limit = None
        self._update_preview()

    @on(Input.Changed, "#output-path-input")
    def on_output_path_changed(self, event: Input.Changed) -> None:
        self.app.config.output_path = event.value or "./logs/"
        self._update_preview()

    @on(Switch.Changed, "#log-samples-switch")
    def on_log_samples_changed(self, event: Switch.Changed) -> None:
        self.app.config.log_samples = event.value
        self._update_preview()

    @on(Select.Changed, "#verbosity-select")
    def on_verbosity_changed(self, event: Select.Changed) -> None:
        self.app.config.verbosity = str(event.value) if event.value else "INFO"
        self._update_preview()

    @on(Button.Pressed, "#run-btn")
    def on_run_button(self) -> None:
        self.action_run()

    @on(Button.Pressed, "#copy-btn")
    def on_copy_button(self) -> None:
        cmd = self._build_command()
        try:
            subprocess.run(["pbcopy"], input=cmd.encode(), check=True)
            self.notify("Command copied to clipboard!")
        except Exception:
            self.notify("Failed to copy to clipboard", severity="error")

    def _update_selected_tasks(self) -> None:
        display = self.query_one("#selected-tasks-display", Static)
        if self.app.config.tasks:
            display.update(", ".join(self.app.config.tasks))
        else:
            display.update("None selected")

    def _update_preview(self) -> None:
        preview = self.query_one("#command-preview", Static)
        cmd = self._build_command()
        preview.update(f"[green]{cmd}[/]")

    def _build_command(self) -> str:
        config = self.app.config
        parts = ["python -m lmms_eval"]
        parts.append(f"--model {config.model}")
        if config.model_args:
            parts.append(f"--model_args {config.model_args}")
        if config.tasks:
            parts.append(f"--tasks {','.join(config.tasks)}")
        parts.append(f"--batch_size {config.batch_size}")
        if config.limit:
            parts.append(f"--limit {config.limit}")
        parts.append(f"--output_path {config.output_path}")
        if config.log_samples:
            parts.append("--log_samples")
        parts.append(f"--verbosity {config.verbosity}")
        if config.device:
            parts.append(f"--device {config.device}")
        return " \\\n    ".join(parts)

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_run(self) -> None:
        if not self.app.config.tasks:
            self.notify("Please select at least one task!", severity="error")
            return
        self.app.push_screen(RunScreen())


class MetricsPlot(Static):
    """Fallback metrics display when textual-plotext is not available."""

    def __init__(self, title: str = "Metrics", **kwargs) -> None:
        super().__init__(**kwargs)
        self.title = title
        self.data: deque = deque(maxlen=50)
        self.timestamps: deque = deque(maxlen=50)

    def add_data_point(self, value: float) -> None:
        self.data.append(value)
        self.timestamps.append(len(self.data))
        self._update_display()

    def _update_display(self) -> None:
        if not self.data:
            self.update(f"[bold]{self.title}[/]: No data yet")
            return

        latest = self.data[-1]
        avg = sum(self.data) / len(self.data)
        max_val = max(self.data)
        min_val = min(self.data)

        # Create a simple ASCII sparkline
        if len(self.data) > 1:
            sparkline = self._create_sparkline(list(self.data))
        else:
            sparkline = "▁"

        self.update(
            f"[bold #5dadec]{self.title}[/bold #5dadec]\n"
            f"[dim]Current:[/] [bold]{latest:.1f}[/] | "
            f"[dim]Avg:[/] {avg:.1f} | "
            f"[dim]Max:[/] {max_val:.1f} | "
            f"[dim]Min:[/] {min_val:.1f}\n"
            f"[#5dadec]{sparkline}[/#5dadec]"
        )

    def _create_sparkline(self, data: list, width: int = 40) -> str:
        """Create an ASCII sparkline from data."""
        if not data:
            return ""
        blocks = "▁▂▃▄▅▆▇█"
        min_val = min(data)
        max_val = max(data)
        val_range = max_val - min_val or 1

        # Resample data if needed
        if len(data) > width:
            step = len(data) / width
            data = [data[int(i * step)] for i in range(width)]

        sparkline = ""
        for val in data:
            idx = int((val - min_val) / val_range * (len(blocks) - 1))
            sparkline += blocks[idx]
        return sparkline

    def clear_data(self) -> None:
        self.data.clear()
        self.timestamps.clear()
        self._update_display()


class SpeedMetricsChart(Container):
    """Container for speed metrics visualization."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.speed_data: deque = deque(maxlen=50)
        self.latency_data: deque = deque(maxlen=50)
        self.tokens_data: deque = deque(maxlen=50)
        self.time_points: deque = deque(maxlen=50)
        self._start_time: float = 0

    def compose(self) -> ComposeResult:
        if HAS_PLOTEXT:
            yield PlotextPlot(id="speed-chart")
            yield PlotextPlot(id="latency-chart")
        else:
            yield MetricsPlot(title="Speed (tokens/s)", id="speed-metrics")
            yield MetricsPlot(title="Latency (s)", id="latency-metrics")

    def on_mount(self) -> None:
        self._start_time = time.time()
        if HAS_PLOTEXT:
            self._init_charts()

    def _init_charts(self) -> None:
        """Initialize the Plotext charts."""
        if not HAS_PLOTEXT:
            return

        # Speed chart
        speed_chart = self.query_one("#speed-chart", PlotextPlot)
        plt = speed_chart.plt
        plt.clear_figure()
        plt.title("Inference Speed (tokens/s)")
        plt.xlabel("Time")
        plt.ylabel("Tokens/s")
        plt.theme("pro")
        speed_chart.refresh()

        # Latency chart
        latency_chart = self.query_one("#latency-chart", PlotextPlot)
        plt = latency_chart.plt
        plt.clear_figure()
        plt.title("Batch Latency (s)")
        plt.xlabel("Time")
        plt.ylabel("Seconds")
        plt.theme("pro")
        latency_chart.refresh()

    def add_metrics(
        self,
        speed: float | None = None,
        latency: float | None = None,
        tokens: int | None = None,
    ) -> None:
        """Add new metrics data points."""
        elapsed = time.time() - self._start_time
        self.time_points.append(elapsed)

        if speed is not None:
            self.speed_data.append(speed)
        if latency is not None:
            self.latency_data.append(latency)
        if tokens is not None:
            self.tokens_data.append(tokens)

        self._update_charts()

    def _update_charts(self) -> None:
        """Update all charts with current data."""
        if HAS_PLOTEXT:
            self._update_plotext_charts()
        else:
            self._update_fallback_charts()

    def _update_plotext_charts(self) -> None:
        """Update Plotext charts."""
        time_list = list(self.time_points)

        # Update speed chart
        if self.speed_data:
            speed_chart = self.query_one("#speed-chart", PlotextPlot)
            plt = speed_chart.plt
            plt.clear_data()
            plt.plot(
                time_list[: len(self.speed_data)],
                list(self.speed_data),
                marker="braille",
            )
            plt.title("Inference Speed (tokens/s)")
            plt.xlabel("Time (s)")
            plt.ylabel("Tokens/s")
            speed_chart.refresh()

        # Update latency chart
        if self.latency_data:
            latency_chart = self.query_one("#latency-chart", PlotextPlot)
            plt = latency_chart.plt
            plt.clear_data()
            plt.plot(
                time_list[: len(self.latency_data)],
                list(self.latency_data),
                marker="braille",
            )
            plt.title("Batch Latency (s)")
            plt.xlabel("Time (s)")
            plt.ylabel("Seconds")
            latency_chart.refresh()

    def _update_fallback_charts(self) -> None:
        """Update fallback ASCII charts."""
        if self.speed_data:
            speed_widget = self.query_one("#speed-metrics", MetricsPlot)
            speed_widget.data = self.speed_data.copy()
            speed_widget._update_display()

        if self.latency_data:
            latency_widget = self.query_one("#latency-metrics", MetricsPlot)
            latency_widget.data = self.latency_data.copy()
            latency_widget._update_display()

    def clear_all(self) -> None:
        """Clear all data."""
        self.speed_data.clear()
        self.latency_data.clear()
        self.tokens_data.clear()
        self.time_points.clear()
        self._start_time = time.time()
        if HAS_PLOTEXT:
            self._init_charts()
        else:
            self.query_one("#speed-metrics", MetricsPlot).clear_data()
            self.query_one("#latency-metrics", MetricsPlot).clear_data()


class RunScreen(Screen):
    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("ctrl+c", "cancel", "Cancel"),
    ]

    def compose(self) -> ComposeResult:
        version = get_version()
        yield Header()
        with Horizontal(id="welcome-header"):
            with Horizontal(id="header-left"):
                yield LogoWidget(id="header-logo")
                with Vertical(id="header-info"):
                    yield Static(
                        f"[bold #5dadec]LMMs-Eval[/bold #5dadec] [dim]v{version}[/dim]"
                    )
                    yield Static(f"[dim]Python {sys.version.split()[0]}[/dim]")
                    yield Static(f"[dim]{LMMS_EVAL_REPO}[/dim]")
            with Vertical(id="header-right"):
                yield Static("[bold]Running Command[/bold]", id="preview-title")
                yield Static("", id="header-command-preview")
        with Container(id="run-container"):
            yield Static("[bold]Running Evaluation...[/]", id="run-title")
            yield Static("", id="run-status")
            yield Rule()
            yield Static("[bold]Performance Metrics[/]", classes="section-title")
            yield SpeedMetricsChart(id="metrics-container")
            yield Rule()
            yield Static("[bold]Command:[/]", classes="section-title")
            yield Static("", id="run-command")
            yield Rule()
            yield Static("[bold]Output:[/]", classes="section-title")
            with VerticalScroll(id="output-scroll"):
                yield Static("", id="run-output")
            with Horizontal():
                yield Button("Run", variant="success", id="execute-btn")
                yield Button("Back", variant="default", id="back-btn")
        yield Footer()

    def on_mount(self) -> None:
        cmd = self._build_command_list()
        self.query_one("#run-command", Static).update(f"[cyan]{' '.join(cmd)}[/]")
        self.query_one("#run-status", Static).update("[yellow]Ready to run[/]")

    def _build_command_list(self) -> list[str]:
        config = self.app.config
        cmd = [sys.executable, "-m", "lmms_eval"]
        cmd.extend(["--model", config.model])
        if config.model_args:
            cmd.extend(["--model_args", config.model_args])
        if config.tasks:
            cmd.extend(["--tasks", ",".join(config.tasks)])
        cmd.extend(["--batch_size", str(config.batch_size)])
        if config.limit:
            cmd.extend(["--limit", str(config.limit)])
        cmd.extend(["--output_path", config.output_path])
        if config.log_samples:
            cmd.append("--log_samples")
        cmd.extend(["--verbosity", config.verbosity])
        if config.device:
            cmd.extend(["--device", config.device])
        return cmd

    def _parse_metrics_from_line(self, line: str) -> dict:
        """Parse speed/latency metrics from log output."""
        import re

        metrics = {}

        # Pattern: "Avg speed: X.X tokens/s" or "avg_speed: X.X"
        speed_patterns = [
            r"[Aa]vg[_\s]?speed[:\s]+(\d+\.?\d*)",
            r"(\d+\.?\d*)\s*tokens?/s",
            r"speed[:\s]+(\d+\.?\d*)",
        ]
        for pattern in speed_patterns:
            match = re.search(pattern, line)
            if match:
                metrics["speed"] = float(match.group(1))
                break

        # Pattern: "latency: X.Xs" or "e2e_latency: X.X"
        latency_patterns = [
            r"[Ll]atency[:\s]+(\d+\.?\d*)",
            r"e2e_latency[:\s]+(\d+\.?\d*)",
            r"[Tt]ime[:\s]+(\d+\.?\d*)s",
        ]
        for pattern in latency_patterns:
            match = re.search(pattern, line)
            if match:
                metrics["latency"] = float(match.group(1))
                break

        # Pattern: "Total tokens: X" or "tokens: X"
        tokens_patterns = [
            r"[Tt]otal[_\s]?tokens[:\s]+(\d+)",
            r"tokens[:\s]+(\d+)",
        ]
        for pattern in tokens_patterns:
            match = re.search(pattern, line)
            if match:
                metrics["tokens"] = int(match.group(1))
                break

        return metrics

    @on(Button.Pressed, "#execute-btn")
    def on_execute(self) -> None:
        self.query_one("#run-status", Static).update("[green]Running...[/]")
        self.query_one("#execute-btn", Button).disabled = True
        # Clear previous metrics
        self.query_one("#metrics-container", SpeedMetricsChart).clear_all()
        self.run_worker(self._run_evaluation())

    @on(Button.Pressed, "#back-btn")
    def on_back(self) -> None:
        self.action_back()

    async def _run_evaluation(self) -> None:
        cmd = self._build_command_list()
        output_widget = self.query_one("#run-output", Static)
        status_widget = self.query_one("#run-status", Static)
        metrics_chart = self.query_one("#metrics-container", SpeedMetricsChart)

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            output_lines = []
            for line in iter(process.stdout.readline, ""):
                output_lines.append(line.rstrip())
                if len(output_lines) > 100:
                    output_lines = output_lines[-100:]
                output_widget.update("\n".join(output_lines))

                # Parse metrics from output
                metrics = self._parse_metrics_from_line(line)
                if metrics:
                    metrics_chart.add_metrics(
                        speed=metrics.get("speed"),
                        latency=metrics.get("latency"),
                        tokens=metrics.get("tokens"),
                    )

            process.wait()

            if process.returncode == 0:
                status_widget.update(
                    "[bold green]Evaluation completed successfully![/]"
                )
            else:
                status_widget.update(
                    f"[bold red]Evaluation failed with code {process.returncode}[/]"
                )
        except Exception as e:
            status_widget.update(f"[bold red]Error: {e}[/]")

        self.query_one("#execute-btn", Button).disabled = False

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_cancel(self) -> None:
        self.app.pop_screen()


class LmmsEvalTUI(App):
    CSS = """
    /* Welcome Screen - top left aligned */
    WelcomeScreen {
        background: #1e1e1e;
    }
    
    WelcomeScreen #welcome-container {
        width: 100%;
        height: 100%;
        align: left top;
        padding: 2 4;
    }
    
    WelcomeScreen #welcome-msg {
        padding: 1 2;
        margin-bottom: 1;
        border: solid #5dadec;
        width: auto;
        background: #2d2d2d;
    }
    
    WelcomeScreen #logo {
        margin: 1 0;
    }
    
    WelcomeScreen #version-info {
        margin-bottom: 1;
    }
    
    WelcomeScreen #continue-msg {
        margin-top: 1;
    }
    
    /* Config/Run Screen Header - two column layout */
    #welcome-header {
        height: auto;
        padding: 1 2;
        background: #2d2d2d;
        border-bottom: solid #5dadec;
    }
    
    #header-left {
        width: 1fr;
        height: auto;
    }
    
    #header-right {
        width: 1fr;
        height: auto;
        padding: 1 2;
        border-left: dashed #5dadec;
    }
    
    #header-logo {
        width: 30;
        height: 15;
        overflow: hidden;
    }
    
    #header-info {
        padding: 1 2;
    }
    
    #preview-title {
        margin-bottom: 1;
    }
    
    #command-preview {
        color: #5dadec;
    }
    
    /* Config Screen */
    #config-container {
        padding: 1;
    }
    
    .section-title {
        margin-top: 1;
        margin-bottom: 1;
    }
    
    .setting-row {
        height: 3;
        margin-bottom: 1;
    }
    
    .setting-label {
        width: 20;
        padding-right: 1;
    }
    
    .setting-input {
        width: 40;
    }
    
    .command-preview {
        background: $surface;
        padding: 1;
        margin: 1;
        border: solid green;
    }
    
    #preview-container {
        height: 100%;
        padding: 1;
    }
    
    #command-scroll {
        height: 1fr;
        min-height: 10;
        margin-bottom: 1;
    }
    
    #preview-buttons {
        dock: bottom;
        height: auto;
        align: center middle;
        padding: 1;
    }
    
    #preview-buttons Button {
        margin: 0 2;
        min-width: 20;
    }
    
    #run-container {
        padding: 2;
    }
    
    #metrics-container {
        height: auto;
        min-height: 8;
        max-height: 16;
        margin: 1;
    }
    
    #speed-chart, #latency-chart {
        height: 8;
        border: solid #5dadec;
        margin: 0 0 1 0;
    }
    
    #speed-metrics, #latency-metrics {
        height: auto;
        min-height: 4;
        border: solid #5dadec;
        padding: 1;
        margin: 0 0 1 0;
    }
    
    #output-scroll {
        height: 15;
        border: solid $primary;
        margin: 1;
    }
    
    Button {
        margin: 1;
    }
    """

    TITLE = "LMMs-Eval TUI"
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("?", "help", "Help"),
    ]

    def __init__(self):
        super().__init__()
        self.config = EvalConfig()

    def on_mount(self) -> None:
        self.push_screen(WelcomeScreen())

    def action_help(self) -> None:
        self.notify("Press Tab to navigate, Enter to select, Escape to go back")


def main():
    app = LmmsEvalTUI()
    app.run()


if __name__ == "__main__":
    main()
