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
    from textual_image.widget import Image as TextualImage

    HAS_TEXTUAL_IMAGE = True
except ImportError:
    HAS_TEXTUAL_IMAGE = False


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


# ASCII art logo for welcome screen
LOGO = r"""
[#87afd7]██╗      ███╗   ███╗███╗   ███╗ ███████╗[/#87afd7]
[#87afd7]██║      ████╗ ████║████╗ ████║██╔════╝ [/#87afd7]
[#87afd7]██║      ██╔████╔██║██╔████╔██║███████╗ [/#87afd7]
[#87afd7]██║      ██║╚██╔╝██║██║╚██╔╝██║╚════██║ [/#87afd7]
[#87afd7]███████╗ ██║ ╚═╝ ██║██║ ╚═╝ ██║███████║ [/#87afd7]
[#5f87af]╚══════╝ ╚═╝     ╚═╝╚═╝     ╚═╝╚══════╝ [/#5f87af]

[#87afd7]███████╗ ██╗   ██╗  █████╗  ██╗         [/#87afd7]
[#87afd7]██╔════╝ ██║   ██║ ██╔══██╗ ██║         [/#87afd7]
[#87afd7]█████╗   ██║   ██║ ███████║ ██║         [/#87afd7]
[#87afd7]██╔══╝   ╚██╗ ██╔╝ ██╔══██║ ██║         [/#87afd7]
[#87afd7]███████╗  ╚████╔╝  ██║  ██║ ███████╗    [/#87afd7]
[#5f87af]╚══════╝   ╚═══╝   ╚═╝  ╚═╝ ╚══════╝    [/#5f87af]
"""


def LogoWidget(**kwargs):
    """Factory function that returns the best available logo widget.

    Uses textual_image (supports TGP/Sixel/Halfcell) if available,
    returns empty widget on unsupported terminals to avoid ugly rendering.
    """
    if HAS_TEXTUAL_IMAGE and os.path.exists(LOGO_IMAGE_PATH):
        return TextualImage(LOGO_IMAGE_PATH, **kwargs)
    # Return empty widget on unsupported terminals
    return Static("", **kwargs)


class WelcomeScreen(Screen):
    BINDINGS = [
        Binding("enter", "start", "Start Configuration"),
        Binding("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        version = get_version()
        yield Container(
            Static(
                "┌────────────────────────────────────┐\n"
                "│        LMMs-Eval Framework         │\n"
                "└────────────────────────────────────┘",
                id="welcome-title",
            ),
            Static(LOGO, id="logo"),
            Static(
                f"v{version}",
                id="version-info",
            ),
            Static("Press Enter to continue", id="continue-msg"),
            Static(f"\n{LMMS_EVAL_REPO}", classes="copyright"),
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
                with Container(id="logo-container"):
                    yield LogoWidget(id="header-logo")
                with Vertical(id="header-info"):
                    yield Static(f"LMMs-Eval SYSTEM", classes="blink")
                    yield Static(f"VER {version}")
                    yield Static(f"PY {sys.version.split()[0]}")
            with Vertical(id="header-right"):
                yield Static("COMMAND PREVIEW", id="preview-title")
                yield Static("", id="command-preview")
        with Container(id="config-container"):
            with TabbedContent():
                with TabPane("MODEL", id="model-tab"):
                    with VerticalScroll():
                        yield Static("MODEL SELECTION", classes="section-title")
                        yield Select(
                            [(name, key) for key, name in POPULAR_MODELS],
                            prompt="Select Model",
                            id="model-select",
                        )
                        yield Rule()
                        yield Static("MODEL ARGUMENTS", classes="section-title")
                        yield Static(
                            "[dim]e.g., model_version=gpt-4o,pretrained=path/to/model[/]"
                        )
                        yield Input(
                            placeholder="Enter model_args (key=value,...)...",
                            id="model-args-input",
                        )
                        yield Rule()
                        yield Static(
                            "API CONFIGURATION",
                            classes="section-title",
                        )
                        yield Static(
                            "[dim]Set OPENAI_API_KEY and OPENAI_API_BASE in environment[/]"
                        )
                        yield Input(
                            placeholder="API BASE URL (OPTIONAL)", id="api-base-input"
                        )
                with TabPane("TASKS", id="tasks-tab"):
                    with VerticalScroll():
                        yield Static("TASK SELECTION", classes="section-title")
                        yield Input(
                            placeholder="Search benchmarks...", id="task-search"
                        )
                        yield Rule()
                        yield Static("AVAILABLE BENCHMARKS", classes="section-title")
                        yield OptionList(
                            *[
                                Option(f"{name} ({key})", id=key)
                                for key, name in POPULAR_TASKS
                            ],
                            id="task-list",
                        )
                        yield Rule()
                        yield Static("SELECTED TASKS", classes="section-title")
                        yield Static("None selected", id="selected-tasks-display")
                with TabPane("SETTINGS", id="settings-tab"):
                    with VerticalScroll():
                        yield Static("SYSTEM CONFIGURATION", classes="section-title")
                        yield Horizontal(
                            Static("BATCH SIZE:", classes="setting-label"),
                            Input(
                                value="1",
                                id="batch-size-input",
                                classes="setting-input",
                            ),
                            classes="setting-row",
                        )
                        yield Horizontal(
                            Static("LIMIT (SAMPLES):", classes="setting-label"),
                            Input(
                                placeholder="ALL (NO LIMIT)",
                                id="limit-input",
                                classes="setting-input",
                            ),
                            classes="setting-row",
                        )
                        yield Horizontal(
                            Static("OUTPUT PATH:", classes="setting-label"),
                            Input(
                                value="./logs/",
                                id="output-path-input",
                                classes="setting-input",
                            ),
                            classes="setting-row",
                        )
                        yield Horizontal(
                            Static("DEVICE:", classes="setting-label"),
                            Input(
                                placeholder="AUTO (CUDA:0, CPU)",
                                id="device-input",
                                classes="setting-input",
                            ),
                            classes="setting-row",
                        )
                        yield Rule()
                        yield Static("LOGGING OPTIONS", classes="section-title")
                        yield Horizontal(
                            Static("LOG SAMPLES:", classes="setting-label"),
                            Switch(value=True, id="log-samples-switch"),
                            classes="setting-row",
                        )
                        yield Horizontal(
                            Static("VERBOSITY:", classes="setting-label"),
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
                with TabPane("RUN", id="run-tab"):
                    with VerticalScroll(id="run-tab-scroll"):
                        yield Static("FINAL CHECK", classes="section-title")
                        yield Static("", id="run-command", classes="command-preview")
                        with Horizontal(id="run-buttons"):
                            yield Button("START", variant="success", id="start-btn")
                            yield Button(
                                "STOP", variant="error", id="stop-btn", disabled=True
                            )
                            yield Button("SAVE CMD", variant="primary", id="copy-btn")
                        yield Static("EVALUATION STATUS", classes="section-title")
                        yield Static("Ready to evaluate", id="run-status")
                        yield Static("OUTPUT LOG", classes="section-title")
                        with VerticalScroll(id="output-scroll"):
                            yield Static("", id="run-output")
        yield Footer()

    def on_mount(self) -> None:
        """Initialize command preview on screen mount."""
        self._update_preview()

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

    _process: subprocess.Popen | None = None

    @on(Button.Pressed, "#start-btn")
    def on_start_button(self) -> None:
        if not self.app.config.tasks:
            self.notify("Please select at least one task!", severity="error")
            return
        self.query_one("#start-btn", Button).disabled = True
        self.query_one("#stop-btn", Button).disabled = False
        self.query_one("#run-status", Static).update("[yellow]Running...[/yellow]")
        self.query_one("#run-output", Static).update("")
        self.run_worker(self._run_evaluation())

    @on(Button.Pressed, "#stop-btn")
    def on_stop_button(self) -> None:
        if self._process:
            self._process.terminate()
            self._process = None
        self.query_one("#start-btn", Button).disabled = False
        self.query_one("#stop-btn", Button).disabled = True
        self.query_one("#run-status", Static).update("[red]Stopped[/red]")

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
        cmd = self._build_command_highlighted()
        try:
            preview = self.query_one("#command-preview", Static)
            preview.update(cmd)
        except Exception:
            pass
        try:
            run_cmd = self.query_one("#run-command", Static)
            run_cmd.update(cmd)
        except Exception:
            pass

    def _build_command_highlighted(self) -> str:
        config = self.app.config
        parts = ["[#87afd7]python -m[/] [bold white]lmms_eval[/]"]
        parts.append(f"[#5f87af]--model[/] [white]{config.model}[/]")
        if config.model_args:
            parts.append(f"[#5f87af]--model_args[/] [white]{config.model_args}[/]")
        if config.tasks:
            parts.append(f"[#5f87af]--tasks[/] [white]{','.join(config.tasks)}[/]")
        parts.append(f"[#5f87af]--batch_size[/] [white]{config.batch_size}[/]")
        if config.limit:
            parts.append(f"[#5f87af]--limit[/] [white]{config.limit}[/]")
        parts.append(f"[#5f87af]--output_path[/] [white]{config.output_path}[/]")
        if config.log_samples:
            parts.append("[#5f87af]--log_samples[/]")
        parts.append(f"[#5f87af]--verbosity[/] [white]{config.verbosity}[/]")
        if config.device:
            parts.append(f"[#5f87af]--device[/] [white]{config.device}[/]")
        return " \\\n    ".join(parts)

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

    async def _run_evaluation(self) -> None:
        cmd = self._build_command_list()
        output_widget = self.query_one("#run-output", Static)
        status_widget = self.query_one("#run-status", Static)

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            output_lines: list[str] = []
            if self._process.stdout:
                for line in iter(self._process.stdout.readline, ""):
                    if self._process is None:
                        break
                    output_lines.append(line.rstrip())
                    if len(output_lines) > 100:
                        output_lines = output_lines[-100:]
                    output_widget.update("\n".join(output_lines))
            if self._process:
                self._process.wait()
                if self._process.returncode == 0:
                    status_widget.update("[bold green]Completed successfully![/]")
                else:
                    status_widget.update(
                        f"[bold red]Failed with code {self._process.returncode}[/]"
                    )
        except Exception as e:
            status_widget.update(f"[bold red]Error: {e}[/]")
        finally:
            self._process = None
            self.query_one("#start-btn", Button).disabled = False
            self.query_one("#stop-btn", Button).disabled = True

    def action_back(self) -> None:
        self.app.pop_screen()


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
            f"[bold #ffd700]{self.title}[/bold #ffd700]\n"
            f"[dim]Current:[/] [bold]{latest:.1f}[/] | "
            f"[dim]Avg:[/] {avg:.1f} | "
            f"[dim]Max:[/] {max_val:.1f} | "
            f"[dim]Min:[/] {min_val:.1f}\n"
            f"[#00cc00]{sparkline}[/#00cc00]"
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
                with Container(id="logo-container"):
                    yield LogoWidget(id="header-logo")
                with Vertical(id="header-info"):
                    yield Static(f"LMMs-Eval SYSTEM", classes="blink")
                    yield Static(f"VER {version}")
                    yield Static(f"PY {sys.version.split()[0]}")
            with Vertical(id="header-right"):
                yield Static("RUNNING COMMAND", id="preview-title")
                yield Static("", id="header-command-preview")
        with Container(id="run-container"):
            yield Static(">>> EVALUATION EXECUTION <<<", id="run-title")
            yield Static("", id="run-status")
            yield Rule()
            yield Static("PERFORMANCE METRICS", classes="section-title")
            yield SpeedMetricsChart(id="metrics-container")
            yield Rule()
            yield Static("COMMAND", classes="section-title")
            yield Static("", id="run-command")
            yield Rule()
            yield Static("OUTPUT LOG", classes="section-title")
            with VerticalScroll(id="output-scroll"):
                yield Static("", id="run-output")
            with Horizontal():
                yield Button("START", variant="success", id="execute-btn")
                yield Button("BACK", variant="default", id="back-btn")
        yield Footer()

    def on_mount(self) -> None:
        cmd_highlighted = self._build_command_highlighted()
        self.query_one("#run-command", Static).update(cmd_highlighted)
        self.query_one("#run-status", Static).update("[yellow]Ready to run[/]")

    def _build_command_highlighted(self) -> str:
        config = self.app.config
        parts = ["[#87afd7]python -m[/] [bold white]lmms_eval[/]"]
        parts.append(f"[#5f87af]--model[/] [white]{config.model}[/]")
        if config.model_args:
            parts.append(f"[#5f87af]--model_args[/] [white]{config.model_args}[/]")
        if config.tasks:
            parts.append(f"[#5f87af]--tasks[/] [white]{','.join(config.tasks)}[/]")
        parts.append(f"[#5f87af]--batch_size[/] [white]{config.batch_size}[/]")
        if config.limit:
            parts.append(f"[#5f87af]--limit[/] [white]{config.limit}[/]")
        parts.append(f"[#5f87af]--output_path[/] [white]{config.output_path}[/]")
        if config.log_samples:
            parts.append("[#5f87af]--log_samples[/]")
        parts.append(f"[#5f87af]--verbosity[/] [white]{config.verbosity}[/]")
        if config.device:
            parts.append(f"[#5f87af]--device[/] [white]{config.device}[/]")
        return " \\\n    ".join(parts)

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
    /* Classic TUI Theme - Soft Blue & White */
    $primary: #5f87af;
    $primary-light: #87afd7;
    $primary-dark: #3a5f7a;
    $accent: #afd7ff;
    $accent-dim: #5f87af;
    $bg-dark: #1c1c1c;
    $bg-medium: #262626;
    $bg-light: #303030;
    $text-bright: #ffffff;
    $text-normal: #d0d0d0;
    $text-dim: #808080;
    $success: #87d787;
    $warning: #d7af5f;
    $error: #d75f5f;
    $surface: #262626;
    $background: #1c1c1c;

    /* Global Screen Styles */
    Screen {
        background: $bg-dark;
        color: $text-normal;
    }

    Tooltip {
        background: $primary;
        color: $text-bright;
        border: solid $accent;
    }

    /* Welcome Screen */
    WelcomeScreen {
        align: center middle;
        background: $bg-dark;
    }
    
    WelcomeScreen #welcome-container {
        width: 70;
        height: auto;
        border: round $primary;
        background: $bg-medium;
        padding: 2 4;
        align: center middle;
    }
    
    WelcomeScreen #welcome-title {
        color: $primary-light;
        text-style: bold;
        content-align: center middle;
        margin-bottom: 1;
    }
    
    WelcomeScreen #logo {
        margin: 1 0;
        content-align: center middle;
        color: $primary;
    }
    
    WelcomeScreen #version-info {
        color: $text-dim;
        content-align: center middle;
        margin-bottom: 2;
    }
    
    WelcomeScreen #continue-msg {
        color: $text-bright;
        text-style: bold;
        content-align: center middle;
        margin-top: 1;
        background: $primary;
        padding: 0 2;
    }

    WelcomeScreen .copyright {
        color: $text-dim;
        content-align: center middle;
        margin-top: 1;
    }
    
    /* Header */
    Header {
        background: $primary-dark;
        color: $text-bright;
        height: 1;
        dock: top;
    }

    #welcome-header {
        height: 10;
        dock: top;
        background: $bg-medium;
        border-bottom: solid $primary;
    }
    
    #header-left {
        width: 40%;
        height: 100%;
        layout: horizontal;
        padding: 1;
        background: $bg-medium;
        border-right: solid $primary-dark;
    }
    
    #header-right {
        width: 60%;
        height: 100%;
        padding: 0;
        background: $bg-dark;
    }
    
    #logo-container {
        width: auto;
        height: 100%;
        margin-right: 2;
    }

    #header-info {
        padding-top: 1;
        color: $text-normal;
    }
    
    #preview-title {
        background: $primary;
        color: $text-bright;
        text-style: bold;
        padding: 0 1;
    }
    
    #command-preview, #header-command-preview {
        color: $accent;
        background: $bg-dark;
        padding: 1;
        height: 1fr;
    }
    
    /* Config Screen Main Area */
    #config-container {
        height: 1fr;
        padding: 0 1;
    }
    
    /* Tabs */
    TabbedContent {
        background: $bg-dark;
    }
    
    TabbedContent > .tabs {
        background: $bg-medium;
        border-bottom: solid $primary;
        padding: 0 1;
    }
    
    Tab {
        background: $bg-light;
        color: $text-dim;
        border: none;
        margin-right: 1;
        padding: 0 2;
    }
    
    Tab:hover {
        color: $text-bright;
        background: $primary-dark;
    }
    
    Tab.-active {
        background: $primary;
        color: $text-bright;
        text-style: bold;
    }
    
    TabPane {
        padding: 1 2;
        background: $bg-medium;
        border: solid $primary-dark;
        border-top: none;
    }
    
    /* Section Titles */
    .section-title {
        color: $primary-light;
        text-style: bold;
        margin: 1 0 0 0;
    }
    
    Rule {
        color: $primary-dark;
    }
    
    /* Inputs & Selects */
    Input {
        background: $bg-dark;
        border: solid $primary-dark;
        color: $text-normal;
    }
    
    Input:focus {
        border: solid $primary;
    }
    
    Select {
        background: $bg-dark;
        border: solid $primary-dark;
        color: $text-normal;
    }
    
    SelectCurrent {
        background: $bg-dark;
        color: $text-normal;
        border: none;
    }

    OptionList {
        background: $bg-dark;
        border: solid $primary-dark;
    }
    
    Switch {
        background: $bg-dark;
        border: solid $primary-dark;
    }
    
    Switch > .switch--slider {
        color: $primary;
    }
    
    Switch.-on > .switch--slider {
        color: $primary-light;
    }
    
    /* Settings Tab Styling */
    .setting-row {
        height: 3;
        margin-bottom: 1;
    }
    
    .setting-label {
        width: 20;
        padding: 1 1 0 0;
        color: $text-normal;
    }
    
    .setting-input {
        width: 1fr;
    }

    /* Run Tab Styling */
    #run-tab {
        padding: 1 2;
    }
    
    #run-tab-scroll {
        height: 1fr;
    }
    
    #run-command {
        background: $bg-dark;
        border: solid $primary-dark;
        padding: 1;
        margin-bottom: 1;
    }
    
    #run-status {
        background: $bg-dark;
        border: solid $primary-dark;
        padding: 0 1;
        color: $text-normal;
        margin-bottom: 1;
    }
    
    #run-output {
        color: $accent;
        background: $bg-dark;
        padding: 1;
    }
    
    #output-scroll {
        border: solid $primary-dark;
        background: $bg-dark;
        height: 1fr;
        min-height: 10;
    }

    /* Buttons - Modern Professional Style */
    Button {
        width: 18;
        height: 3;
        border: tall $bg-medium;
        background: $bg-dark;
        color: $text-normal;
        text-style: bold;
        content-align: center middle;
    }
    
    Button:hover {
        border: tall $primary-light;
        color: $primary-light;
    }
    
    /* Primary (Save) - Muted Blue Outline */
    Button.-primary {
        border: tall $primary-dark;
        color: $primary-light;
    }
    
    Button.-primary:hover {
        background: $primary-dark;
        color: $text-bright;
        border: tall $primary-dark;
    }
    
    /* Success (Start) - Muted Green Outline */
    Button.-success {
        border: tall #4e704e;
        color: #87d787;
    }
    
    Button.-success:hover {
        background: #4e704e;
        color: $text-bright;
        border: tall #4e704e;
    }
    
    /* Error (Stop) - Muted Red Outline */
    Button.-error {
        border: tall #8a4b4b;
        color: #d75f5f;
    }
    
    Button.-error:hover {
        background: #8a4b4b;
        color: $text-bright;
        border: tall #8a4b4b;
    }
    
    #run-buttons {
        height: auto;
        padding: 1 0;
        align: center middle;
        layout: horizontal;
    }
    
    #run-buttons Button {
        margin: 0 1;
    }

    /* Charts (RunScreen) */
    #metrics-container {
        height: auto;
        min-height: 10;
        background: $bg-dark;
        border: solid $primary-dark;
        padding: 1;
    }
    
    #speed-chart, #latency-chart {
        height: 10;
        border: solid $primary-dark;
        background: $bg-dark;
        margin-bottom: 1;
    }
    
    #speed-metrics, #latency-metrics {
        height: auto;
        border: solid $primary-dark;
        background: $bg-dark;
        padding: 1;
        margin-bottom: 1;
    }
    
    /* Footer */
    Footer {
        background: $bg-medium;
        color: $text-dim;
    }
    
    Footer > .footer--key {
        background: $primary-dark;
        color: $text-bright;
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
