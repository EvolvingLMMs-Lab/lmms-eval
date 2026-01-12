"""
LMMs-Eval TUI Application
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import ClassVar

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
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


class WelcomeScreen(Screen):
    BINDINGS = [
        Binding("enter", "start", "Start Configuration"),
        Binding("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static(
                """
[bold cyan]╔═══════════════════════════════════════════════════════════════╗[/]
[bold cyan]║[/]                                                               [bold cyan]║[/]
[bold cyan]║[/]   [bold white]LMMs-Eval TUI[/]                                              [bold cyan]║[/]
[bold cyan]║[/]   [dim]Interactive evaluation for Large Multimodal Models[/]        [bold cyan]║[/]
[bold cyan]║[/]                                                               [bold cyan]║[/]
[bold cyan]╚═══════════════════════════════════════════════════════════════╝[/]

[bold green]Features:[/]
  • Select models and tasks interactively
  • Configure evaluation parameters
  • Preview and run evaluations
  • Real-time progress tracking

[bold yellow]Quick Start:[/]
  Press [bold]Enter[/] to begin configuration
  Press [bold]q[/] to quit

[dim]Version: 0.5.0 | https://github.com/EvolvingLMMs-Lab/lmms-eval[/]
""",
                id="welcome-text",
            ),
            Button("Start Configuration", variant="primary", id="start-btn"),
            id="welcome-container",
        )
        yield Footer()

    def action_start(self) -> None:
        self.app.push_screen(ConfigScreen())

    def action_quit(self) -> None:
        self.app.exit()

    @on(Button.Pressed, "#start-btn")
    def on_start_button(self) -> None:
        self.action_start()


class ConfigScreen(Screen):
    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("ctrl+r", "run", "Run Evaluation"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            TabbedContent(
                TabPane("Model", id="model-tab"),
                TabPane("Tasks", id="tasks-tab"),
                TabPane("Settings", id="settings-tab"),
                TabPane("Preview", id="preview-tab"),
            ),
            id="config-container",
        )
        yield Footer()

    def on_mount(self) -> None:
        self._setup_model_tab()
        self._setup_tasks_tab()
        self._setup_settings_tab()
        self._setup_preview_tab()

    def _setup_model_tab(self) -> None:
        tab = self.query_one("#model-tab", TabPane)
        tab.mount(
            VerticalScroll(
                Static("[bold]Select Model[/]", classes="section-title"),
                Select(
                    [(name, key) for key, name in POPULAR_MODELS],
                    prompt="Choose a model",
                    id="model-select",
                ),
                Rule(),
                Static("[bold]Model Arguments[/]", classes="section-title"),
                Static("[dim]e.g., model_version=gpt-4o,pretrained=path/to/model[/]"),
                Input(
                    placeholder="model_args (comma-separated key=value pairs)",
                    id="model-args-input",
                ),
                Rule(),
                Static(
                    "[bold]API Configuration (for API models)[/]",
                    classes="section-title",
                ),
                Static("[dim]Set OPENAI_API_KEY and OPENAI_API_BASE in environment[/]"),
                Input(placeholder="API Base URL (optional)", id="api-base-input"),
            )
        )

    def _setup_tasks_tab(self) -> None:
        tab = self.query_one("#tasks-tab", TabPane)
        tab.mount(
            VerticalScroll(
                Static("[bold]Search Tasks[/]", classes="section-title"),
                Input(placeholder="Type to filter tasks...", id="task-search"),
                Rule(),
                Static("[bold]Popular Tasks[/]", classes="section-title"),
                OptionList(
                    *[Option(f"{name} ({key})", id=key) for key, name in POPULAR_TASKS],
                    id="task-list",
                ),
                Rule(),
                Static("[bold]Selected Tasks[/]", classes="section-title"),
                Static("None selected", id="selected-tasks-display"),
            )
        )

    def _setup_settings_tab(self) -> None:
        tab = self.query_one("#settings-tab", TabPane)
        tab.mount(
            VerticalScroll(
                Static("[bold]Evaluation Settings[/]", classes="section-title"),
                Horizontal(
                    Static("Batch Size:", classes="setting-label"),
                    Input(value="1", id="batch-size-input", classes="setting-input"),
                    classes="setting-row",
                ),
                Horizontal(
                    Static("Limit (samples):", classes="setting-label"),
                    Input(
                        placeholder="None (all samples)",
                        id="limit-input",
                        classes="setting-input",
                    ),
                    classes="setting-row",
                ),
                Horizontal(
                    Static("Output Path:", classes="setting-label"),
                    Input(
                        value="./logs/", id="output-path-input", classes="setting-input"
                    ),
                    classes="setting-row",
                ),
                Horizontal(
                    Static("Device:", classes="setting-label"),
                    Input(
                        placeholder="auto (cuda:0, cpu, etc.)",
                        id="device-input",
                        classes="setting-input",
                    ),
                    classes="setting-row",
                ),
                Rule(),
                Static("[bold]Options[/]", classes="section-title"),
                Horizontal(
                    Static("Log Samples:", classes="setting-label"),
                    Switch(value=True, id="log-samples-switch"),
                    classes="setting-row",
                ),
                Horizontal(
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
                ),
            )
        )

    def _setup_preview_tab(self) -> None:
        tab = self.query_one("#preview-tab", TabPane)
        tab.mount(
            VerticalScroll(
                Static("[bold]Command Preview[/]", classes="section-title"),
                Static("", id="command-preview", classes="command-preview"),
                Rule(),
                Button("Run Evaluation", variant="success", id="run-btn"),
                Button("Copy Command", variant="primary", id="copy-btn"),
            )
        )

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


class RunScreen(Screen):
    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("ctrl+c", "cancel", "Cancel"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static("[bold]Running Evaluation...[/]", id="run-title"),
            Static("", id="run-status"),
            Rule(),
            Static("[bold]Command:[/]", classes="section-title"),
            Static("", id="run-command"),
            Rule(),
            Static("[bold]Output:[/]", classes="section-title"),
            VerticalScroll(
                Static("", id="run-output"),
                id="output-scroll",
            ),
            Horizontal(
                Button("Run", variant="success", id="execute-btn"),
                Button("Back", variant="default", id="back-btn"),
            ),
            id="run-container",
        )
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

    @on(Button.Pressed, "#execute-btn")
    def on_execute(self) -> None:
        self.query_one("#run-status", Static).update("[green]Running...[/]")
        self.query_one("#execute-btn", Button).disabled = True
        self.run_worker(self._run_evaluation())

    @on(Button.Pressed, "#back-btn")
    def on_back(self) -> None:
        self.action_back()

    async def _run_evaluation(self) -> None:
        cmd = self._build_command_list()
        output_widget = self.query_one("#run-output", Static)
        status_widget = self.query_one("#run-status", Static)

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
    #welcome-container {
        align: center middle;
        padding: 2;
    }
    
    #welcome-text {
        text-align: center;
        margin-bottom: 2;
    }
    
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
    
    #run-container {
        padding: 2;
    }
    
    #output-scroll {
        height: 20;
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
