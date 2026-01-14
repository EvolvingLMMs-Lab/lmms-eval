"""
LMMs-Eval TUI Application
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from dataclasses import dataclass, field
from importlib.metadata import version as pkg_version

from rich.markup import escape as rich_escape
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.timer import Timer
from textual.widgets import (
    Button,
    ContentSwitcher,
    Footer,
    Header,
    Input,
    OptionList,
    Rule,
    Select,
    Static,
    TextArea,
)
from textual.widgets.option_list import Option

from lmms_eval.tui.discovery import get_discovery_cache

TAB_IDS = ["model-tab", "tasks-tab", "settings-tab"]
TAB_LABELS = ["MODEL", "TASKS", "SETTINGS"]

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


LOGO_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "assets", "logo.jpg")

_ANSI_ESCAPE_RE = None


def _strip_ansi(text: str) -> str:
    global _ANSI_ESCAPE_RE
    if _ANSI_ESCAPE_RE is None:
        import re

        _ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
    return _ANSI_ESCAPE_RE.sub("", text)


def _highlight_bash_output(text: str) -> str:
    import re

    lines = text.split("\n")
    highlighted = []
    for line in lines:
        if re.search(r"\b(error|Error|ERROR|failed|Failed|FAILED|exception)\b", line):
            highlighted.append(f"[#f87171]{line}[/]")
        elif re.search(r"\b(warning|Warning|WARNING|warn|Warn|WARN)\b", line):
            highlighted.append(f"[#e5a84b]{line}[/]")
        elif re.search(
            r"\b(success|Success|SUCCESS|passed|Passed|PASSED|done|Done|DONE|completed|Completed)\b",
            line,
        ):
            highlighted.append(f"[#4ec9b0]{line}[/]")
        elif re.search(r"\b(INFO|DEBUG|info|debug)\b", line):
            highlighted.append(f"[#7a9eb5]{line}[/]")
        elif re.search(r"\d+%|\[\d+/\d+\]|━|█|▓|░", line):
            highlighted.append(f"[#6ed7c5]{line}[/]")
        else:
            highlighted.append(f"[#d0d0d0]{line}[/]")
    return "\n".join(highlighted)


def _get_hostname() -> str:
    import socket

    try:
        return socket.gethostname().split(".")[0]
    except Exception:
        return "unknown"


def _get_git_branch() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return ""


def _get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return ""


_CACHED_DEVICE: str | None = None


def _detect_device() -> str:
    global _CACHED_DEVICE
    if _CACHED_DEVICE is not None:
        return _CACHED_DEVICE
    try:
        import torch

        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            _CACHED_DEVICE = "cuda:0" if count == 1 else f"cuda:0 - cuda:{count - 1}"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            _CACHED_DEVICE = "mps"
        else:
            _CACHED_DEVICE = "cpu"
    except ImportError:
        _CACHED_DEVICE = "cpu"
    return _CACHED_DEVICE


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
    model_args: str = "model_version=allenai/molmo-2-8b:free"
    tasks: list[str] = field(default_factory=lambda: ["mme"])
    batch_size: int = 1
    limit: int | None = 10
    output_path: str = "./logs/"
    log_samples: bool = True
    verbosity: str = "INFO"
    device: str | None = None
    num_fewshot: int | None = None
    api_env: str = 'export OPENAI_API_KEY="${OPENROUTER_API_KEY}"\nexport OPENAI_API_BASE="https://openrouter.ai/api/v1"'
    activate_cmd: str = "source .venv/bin/activate"


def _get_models() -> list[tuple[str, str]]:
    return get_discovery_cache().get_models()


def _get_tasks() -> list[tuple[str, str]]:
    return get_discovery_cache().get_tasks()


LOGO = r"""
[#4ec9b0]██╗      ███╗   ███╗███╗   ███╗ ███████╗[/#4ec9b0]
[#4ec9b0]██║      ████╗ ████║████╗ ████║██╔════╝ [/#4ec9b0]
[#4ec9b0]██║      ██╔████╔██║██╔████╔██║███████╗ [/#4ec9b0]
[#4ec9b0]██║      ██║╚██╔╝██║██║╚██╔╝██║╚════██║ [/#4ec9b0]
[#4ec9b0]███████╗ ██║ ╚═╝ ██║██║ ╚═╝ ██║███████╗ [/#4ec9b0]
[#3aa08a]╚══════╝ ╚═╝     ╚═╝╚═╝     ╚═╝╚══════╝ [/#3aa08a]

[#4ec9b0]███████╗ ██╗   ██╗  █████╗  ██╗         [/#4ec9b0]
[#4ec9b0]██╔════╝ ██║   ██║ ██╔══██╗ ██║         [/#4ec9b0]
[#4ec9b0]█████╗   ██║   ██║ ███████║ ██║         [/#4ec9b0]
[#4ec9b0]██╔══╝   ╚██╗ ██╔╝ ██╔══██║ ██║         [/#4ec9b0]
[#4ec9b0]███████╗  ╚████╔╝  ██║  ██║ ███████╗    [/#4ec9b0]
[#3aa08a]╚══════╝   ╚═══╝   ╚═╝  ╚═╝ ╚══════╝    [/#3aa08a]
"""

LOGO_SMALL = r"""[#4ec9b0]██╗     ███╗   ███╗███╗   ███╗███████╗[/#4ec9b0]
[#4ec9b0]██║     ████╗ ████║████╗ ████║██╔════╝[/#4ec9b0]
[#4ec9b0]██║     ██╔████╔██║██╔████╔██║███████╗[/#4ec9b0]
[#4ec9b0]██║     ██║╚██╔╝██║██║╚██╔╝██║╚════██║[/#4ec9b0]
[#4ec9b0]███████╗██║ ╚═╝ ██║██║ ╚═╝ ██║███████║[/#4ec9b0]
[#3aa08a]╚══════╝╚═╝     ╚═╝╚═╝     ╚═╝╚══════╝[/#3aa08a]"""


def _terminal_supports_images() -> bool:
    """Detect terminals with image protocol support (iTerm2/Kitty/WezTerm/Ghostty/Sixel)."""
    term_program = os.environ.get("TERM_PROGRAM", "").lower()
    term = os.environ.get("TERM", "").lower()

    if term_program in ("iterm.app", "wezterm", "mintty", "ghostty"):
        return True
    if "kitty" in term or os.environ.get("KITTY_WINDOW_ID"):
        return True
    if "ghostty" in term or os.environ.get("GHOSTTY_RESOURCES_DIR"):
        return True
    if os.environ.get("SIXEL_SUPPORT") == "1":
        return True
    return False


def LogoWidget(**kwargs):
    if HAS_TEXTUAL_IMAGE and os.path.exists(LOGO_IMAGE_PATH):
        return TextualImage(LOGO_IMAGE_PATH, **kwargs)
    return Static("", **kwargs)


class WelcomeScreen(Screen):
    BINDINGS = [
        Binding("q", "quit", "Quit"),
    ]

    AUTO_START_SECONDS = 5
    _auto_timer: Timer | None = None
    _countdown: int = 5

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
            Static(
                "Starting in 5s... [dim](press any key to skip)[/dim]",
                id="continue-msg",
            ),
            Static(
                f"[dim]★ Star us on GitHub:[/dim] {LMMS_EVAL_REPO}\n"
                f"[dim]✎ Issues & Feedback:[/dim] {LMMS_EVAL_ISSUES}",
                classes="copyright",
            ),
            id="welcome-container",
        )

    def on_mount(self) -> None:
        self._countdown = self.AUTO_START_SECONDS
        self._auto_timer = self.set_interval(1.0, self._tick_countdown)

    def _tick_countdown(self) -> None:
        self._countdown -= 1
        if self._countdown <= 0:
            self._stop_timer()
            self.action_start()
        else:
            try:
                msg = self.query_one("#continue-msg", Static)
                msg.update(
                    f"Starting in {self._countdown}s... [dim](press any key to skip)[/dim]"
                )
            except Exception:
                pass

    def _stop_timer(self) -> None:
        if self._auto_timer is not None:
            self._auto_timer.stop()
            self._auto_timer = None

    def on_click(self) -> None:
        self.action_start()

    def action_start(self) -> None:
        self._stop_timer()
        self.app.push_screen(ConfigScreen())

    def action_quit(self) -> None:
        self._stop_timer()
        self.app.exit()

    def on_key(self, event) -> None:
        if event.key == "q":
            self.action_quit()
        else:
            self.action_start()


class ConfigScreen(Screen):
    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("ctrl+r", "reload", "Reload Tasks/Models"),
    ]

    _preview_timer: Timer | None = None
    PREVIEW_DEBOUNCE_MS = 50
    _loaded_tabs: set[str]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._loaded_tabs = {"model-tab"}

    def action_reload(self) -> None:
        cache = get_discovery_cache()
        task_count, model_count = cache.reload()
        self._refresh_model_select()
        self._refresh_task_list()
        self.notify(f"Reloaded {task_count} tasks, {model_count} models")

    def _refresh_model_select(self) -> None:
        try:
            model_select = self.query_one("#model-select", Select)
            current_value = model_select.value
            model_select.set_options([(name, key) for key, name in _get_models()])
            if current_value:
                model_select.value = current_value
        except Exception:
            pass

    def _refresh_task_list(self) -> None:
        try:
            task_list = self.query_one("#task-list", OptionList)
            task_list.clear_options()
            for key, name in _get_tasks():
                task_list.add_option(Option(f"{name} ({key})", id=key))
        except Exception:
            pass

    def _schedule_preview_update(self) -> None:
        if self._preview_timer is not None:
            self._preview_timer.stop()
        self._preview_timer = self.set_timer(
            self.PREVIEW_DEBOUNCE_MS / 1000.0, self._update_preview
        )

    def compose(self) -> ComposeResult:
        version = get_version()
        yield Header()
        with Horizontal(id="header-split"):
            with Vertical(id="header-left"):
                yield Static("LMMs-Eval Terminal UI", id="header-title")
                with Horizontal(id="header-left-content"):
                    with Container(id="logo-container"):
                        yield LogoWidget(id="header-image")
                    with Vertical(id="header-info"):
                        hostname = _get_hostname()
                        cwd = os.path.basename(os.getcwd())
                        yield Static(f"[dim]host[/]   {hostname}")
                        yield Static(f"[dim]dir[/]    {cwd}")
                        yield Static("[dim]branch[/] ...", id="git-branch-info")
                        yield Static("[dim]commit[/] ...", id="git-commit-info")
                        yield Static(f"[dim]ver[/]    {version}")
            with Vertical(id="header-right"):
                yield Static("COMMAND PREVIEW", id="preview-title")
                with VerticalScroll(id="command-preview-scroll"):
                    yield Static("", id="run-command")
        with Container(id="config-container"):
            with Horizontal(id="top-bar"):
                with Horizontal(id="custom-tabs"):
                    for i, label in enumerate(TAB_LABELS):
                        classes = "custom-tab"
                        if i == 0:
                            classes += " custom-tab-active"
                        yield Button(label, id=f"tab-btn-{TAB_IDS[i]}", classes=classes)
                with Horizontal(id="run-buttons"):
                    yield Button("START", variant="success", id="start-btn")
                    yield Button("STOP", variant="error", id="stop-btn", disabled=True)
                    yield Button("SAVE CMD", variant="primary", id="copy-btn")
            with Horizontal(id="main-split"):
                with Vertical(id="left-panel"):
                    with ContentSwitcher(initial="model-tab", id="tab-content"):
                        with Container(id="model-tab"):
                            with VerticalScroll(id="model-tab-content"):
                                yield Static(
                                    "MODEL SELECTION", classes="section-title-first"
                                )
                                yield Select(
                                    [(name, key) for key, name in _get_models()],
                                    value="openai_compatible",
                                    id="model-select",
                                )
                                yield Static("MODEL ARGUMENTS", classes="section-title")
                                yield Static(
                                    "[dim]> e.g., model_version=gpt-4o,pretrained=path/to/model[/]",
                                    classes="hint-text",
                                )
                                yield Input(
                                    value="model_version=allenai/molmo-2-8b:free",
                                    id="model-args-input",
                                )
                                yield Static(
                                    "ACTIVATE ENVIRONMENT",
                                    classes="section-title",
                                )
                                yield Static(
                                    "[dim]> Command to activate env (e.g., source .venv/bin/activate, conda activate myenv)[/]",
                                    classes="hint-text",
                                )
                                yield Input(
                                    value="source .venv/bin/activate",
                                    id="activate-cmd-input",
                                )
                                yield Static(
                                    "API CONFIGURATION",
                                    classes="section-title",
                                )
                                yield Static(
                                    "[dim]> Environment variables for API access (edit below or set in shell)[/]",
                                    classes="hint-text",
                                )
                                yield TextArea(
                                    'export OPENAI_API_KEY="${OPENROUTER_API_KEY}"\nexport OPENAI_API_BASE="https://openrouter.ai/api/v1"',
                                    id="api-env-input",
                                    classes="api-env-textarea",
                                )
                        with Container(id="tasks-tab"):
                            yield Static(
                                "[dim]Loading...[/dim]",
                                id="tasks-tab-placeholder",
                                classes="tab-placeholder",
                            )
                        with Container(id="settings-tab"):
                            yield Static(
                                "[dim]Loading...[/dim]",
                                id="settings-tab-placeholder",
                                classes="tab-placeholder",
                            )
                with Vertical(id="right-panel"):
                    yield Static("OUTPUT LOG", classes="section-title-first")
                    yield Static("Ready to evaluate", id="run-status")
                    with VerticalScroll(id="output-scroll"):
                        yield Static("", id="run-output")
        yield Footer()

    def on_mount(self) -> None:
        self.set_timer(0.05, self._deferred_init)

    def _deferred_init(self) -> None:
        self._update_preview()
        self._load_git_info()
        self._load_device_info()

    @work(thread=True)
    def _load_git_info(self) -> None:
        git_branch = _get_git_branch()
        git_commit = _get_git_commit()
        self.app.call_from_thread(self._apply_git_info, git_branch, git_commit)

    @work(thread=True)
    def _load_device_info(self) -> None:
        device = _detect_device()
        self.app.call_from_thread(self._apply_device_info, device)

    def _apply_device_info(self, device: str) -> None:
        try:
            device_input = self.query_one("#device-input", Input)
            device_input.placeholder = f"auto ({device})"
        except Exception:
            pass

    def _apply_git_info(self, branch: str, commit: str) -> None:
        try:
            branch_widget = self.query_one("#git-branch-info", Static)
            commit_widget = self.query_one("#git-commit-info", Static)
            if branch:
                branch_widget.update(f"[dim]branch[/] {branch}")
            else:
                branch_widget.display = False
            if commit:
                commit_widget.update(f"[dim]commit[/] {commit}")
            else:
                commit_widget.display = False
        except Exception:
            pass

    @on(Button.Pressed, ".custom-tab")
    def on_custom_tab_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""
        if not btn_id.startswith("tab-btn-"):
            return
        tab_id = btn_id.replace("tab-btn-", "")
        switcher = self.query_one("#tab-content", ContentSwitcher)
        switcher.current = tab_id
        for btn in self.query(".custom-tab"):
            btn.remove_class("custom-tab-active")
        event.button.add_class("custom-tab-active")
        if tab_id not in self._loaded_tabs:
            self._loaded_tabs.add(tab_id)
            self._populate_tab(tab_id)

    def _populate_tab(self, tab_id: str) -> None:
        tab_container = self.query_one(f"#{tab_id}", Container)

        if tab_id == "tasks-tab":
            self._populate_tasks_tab(tab_container)
        elif tab_id == "settings-tab":
            self._populate_settings_tab(tab_container)

    def _populate_tasks_tab(self, tab_container: Container) -> None:
        try:
            placeholder = tab_container.query_one("#tasks-tab-placeholder")
            placeholder.remove()
        except Exception:
            pass

        scroll = VerticalScroll()
        tab_container.mount(scroll)
        scroll.mount(Static("TASK SELECTION", classes="section-title-first"))
        scroll.mount(Input(placeholder="Search benchmarks...", id="task-search"))
        scroll.mount(Static("AVAILABLE BENCHMARKS", classes="section-title"))
        scroll.mount(
            OptionList(
                *[Option(f"{name} ({key})", id=key) for key, name in _get_tasks()],
                id="task-list",
            )
        )
        scroll.mount(Static("SELECTED TASKS", classes="section-title"))
        scroll.mount(Horizontal(id="selected-tasks-container"))
        self._update_selected_tasks()

    def _populate_settings_tab(self, tab_container: Container) -> None:
        try:
            placeholder = tab_container.query_one("#settings-tab-placeholder")
            placeholder.remove()
        except Exception:
            pass

        scroll = VerticalScroll()
        tab_container.mount(scroll)
        scroll.mount(Static("SYSTEM CONFIGURATION", classes="section-title-first"))
        scroll.mount(
            Horizontal(
                Static("batch_size:", classes="setting-label"),
                Input(value="1", id="batch-size-input", classes="setting-input"),
                classes="setting-row",
            )
        )
        scroll.mount(
            Horizontal(
                Static("limit:", classes="setting-label"),
                Input(
                    value="10",
                    placeholder="ALL (NO LIMIT)",
                    id="limit-input",
                    classes="setting-input",
                ),
                classes="setting-row",
            )
        )
        scroll.mount(
            Horizontal(
                Static("output_path:", classes="setting-label"),
                Input(value="./logs/", id="output-path-input", classes="setting-input"),
                classes="setting-row",
            )
        )
        device = _CACHED_DEVICE or "auto"
        scroll.mount(
            Horizontal(
                Static("device:", classes="setting-label"),
                Input(
                    placeholder=f"auto ({device})",
                    id="device-input",
                    classes="setting-input",
                ),
                classes="setting-row setting-row-last",
            )
        )
        scroll.mount(Static("LOGGING OPTIONS", classes="section-title"))
        scroll.mount(
            Horizontal(
                Static("LOG SAMPLES:", classes="setting-label"),
                Select(
                    [("YES", True), ("NO", False)],
                    value=True,
                    id="log-samples-select",
                    classes="setting-input",
                ),
                classes="setting-row",
            )
        )
        scroll.mount(
            Horizontal(
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
                    classes="setting-input",
                ),
                classes="setting-row setting-row-last",
            )
        )

    @on(Select.Changed, "#model-select")
    def on_model_changed(self, event: Select.Changed) -> None:
        self.app.config.model = str(event.value) if event.value else "openai_compatible"
        self._update_preview()

    @on(Input.Changed, "#model-args-input")
    def on_model_args_changed(self, event: Input.Changed) -> None:
        self.app.config.model_args = event.value
        self._schedule_preview_update()

    @on(Input.Changed, "#activate-cmd-input")
    def on_activate_cmd_changed(self, event: Input.Changed) -> None:
        self.app.config.activate_cmd = event.value or ""
        self._schedule_preview_update()

    @on(OptionList.OptionSelected, "#task-list")
    def on_task_selected(self, event: OptionList.OptionSelected) -> None:
        task_id = str(event.option.id)
        if task_id not in self.app.config.tasks:
            self.app.config.tasks.append(task_id)
        self._update_selected_tasks()
        self._update_preview()

    @on(Button.Pressed, ".task-card")
    def on_task_card_pressed(self, event: Button.Pressed) -> None:
        task_id = getattr(event.button, "_task_id", None)
        if task_id and task_id in self.app.config.tasks:
            self.app.config.tasks.remove(task_id)
            self._update_selected_tasks()
            self._update_preview()

    @on(Input.Changed, "#batch-size-input")
    def on_batch_size_changed(self, event: Input.Changed) -> None:
        try:
            self.app.config.batch_size = int(event.value) if event.value else 1
        except ValueError:
            pass
        self._schedule_preview_update()

    @on(Input.Changed, "#limit-input")
    def on_limit_changed(self, event: Input.Changed) -> None:
        try:
            self.app.config.limit = int(event.value) if event.value else None
        except ValueError:
            self.app.config.limit = None
        self._schedule_preview_update()

    @on(Input.Changed, "#output-path-input")
    def on_output_path_changed(self, event: Input.Changed) -> None:
        self.app.config.output_path = event.value or "./logs/"
        self._schedule_preview_update()

    @on(Select.Changed, "#log-samples-select")
    def on_log_samples_changed(self, event: Select.Changed) -> None:
        self.app.config.log_samples = bool(event.value)
        self._update_preview()

    @on(Select.Changed, "#verbosity-select")
    def on_verbosity_changed(self, event: Select.Changed) -> None:
        self.app.config.verbosity = str(event.value) if event.value else "INFO"
        self._update_preview()

    @on(TextArea.Changed, "#api-env-input")
    def on_api_env_changed(self, event: TextArea.Changed) -> None:
        self.app.config.api_env = event.text_area.text
        self._schedule_preview_update()

    _process: subprocess.Popen | None = None
    _stop_requested: bool = False
    _loading_timer: Timer | None = None
    _loading_frame: int = 0
    _has_output: bool = False
    LOADING_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def _start_loading_animation(self) -> None:
        self._has_output = False
        self._loading_frame = 0
        self._loading_timer = self.set_interval(0.1, self._tick_loading)

    def _tick_loading(self) -> None:
        if self._has_output:
            self._stop_loading_animation()
            return
        self._loading_frame = (self._loading_frame + 1) % len(self.LOADING_FRAMES)
        spinner = self.LOADING_FRAMES[self._loading_frame]
        try:
            self.query_one("#run-output", Static).update(
                f"[#87afd7]{spinner}[/] [dim]Initializing evaluation pipeline...[/dim]"
            )
        except Exception:
            pass

    def _stop_loading_animation(self) -> None:
        if self._loading_timer is not None:
            self._loading_timer.stop()
            self._loading_timer = None

    @on(Button.Pressed, "#start-btn")
    def on_start_button(self) -> None:
        if not self.app.config.tasks:
            self.notify("Please select at least one task!", severity="error")
            return
        self.query_one("#start-btn", Button).disabled = True
        self.query_one("#stop-btn", Button).disabled = False
        self.query_one("#run-status", Static).update("[yellow]Running...[/yellow]")
        self.query_one("#run-output", Static).update("")
        self._start_loading_animation()
        self._run_evaluation_worker()

    @on(Button.Pressed, "#stop-btn")
    def on_stop_button(self) -> None:
        self._stop_loading_animation()
        self._stop_requested = True
        pid = None
        if self._process:
            pid = self._process.pid
            try:
                if self._process.stdout:
                    self._process.stdout.close()
            except Exception:
                pass
            try:
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
            except (ProcessLookupError, OSError):
                try:
                    self._process.terminate()
                except Exception:
                    pass
            try:
                self._process.kill()
            except Exception:
                pass

        self._append_termination_message(pid)
        self.query_one("#start-btn", Button).disabled = False
        self.query_one("#stop-btn", Button).disabled = True
        self.query_one("#run-status", Static).update("[bold red]⏹ Stopped[/]")

    def _append_termination_message(self, pid: int | None) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        try:
            output_widget = self.query_one("#run-output", Static)
            current_output = output_widget.renderable
            current_text = str(current_output) if current_output else ""

            separator = "─" * 50
            kill_msg = f"\n\n[bold red]{separator}[/]\n"
            kill_msg += f"[bold red]⏹ TERMINATED[/] [dim]at {timestamp}[/]\n"
            if pid:
                kill_msg += f"[dim]Process PID: {pid} killed with SIGTERM[/]\n"
            kill_msg += f"[bold red]{separator}[/]"

            output_widget.update(current_text + kill_msg)
        except Exception:
            pass

    @on(Button.Pressed, "#copy-btn")
    def on_copy_button(self) -> None:
        cmd = self._build_command()
        try:
            subprocess.run(["pbcopy"], input=cmd.encode(), check=True)
            self.notify("Command copied to clipboard!")
        except Exception:
            self.notify("Failed to copy to clipboard", severity="error")

    def _update_selected_tasks(self) -> None:
        try:
            container = self.query_one("#selected-tasks-container", Horizontal)
        except Exception:
            return
        with self.app.batch_update():
            for child in list(container.children):
                child.remove()
            if self.app.config.tasks:
                for task_id in self.app.config.tasks:
                    task_name = next(
                        (name for key, name in _get_tasks() if key == task_id),
                        task_id,
                    )
                    card = Button(f"✕ {task_id}", classes="task-card")
                    card.tooltip = task_name
                    card._task_id = task_id
                    container.mount(card)
            else:
                container.mount(Static("No tasks selected", classes="no-tasks-msg"))

    def _update_preview(self) -> None:
        cmd_full = self._build_command_highlighted_full()
        try:
            run_cmd = self.query_one("#run-command", Static)
            run_cmd.update(cmd_full)
        except Exception:
            pass

    def _build_command_highlighted_full(self) -> str:
        config = self.app.config
        lines = []
        if config.api_env and config.api_env.strip():
            for env_line in config.api_env.strip().split("\n"):
                if env_line.strip():
                    lines.append(f"[dim]{env_line}[/dim]")
            lines.append("")
        if config.activate_cmd and config.activate_cmd.strip():
            lines.append(f"[dim]{config.activate_cmd}[/dim]")
            lines.append("")
        lines.append(self._build_command_highlighted())
        return "\n".join(lines)

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
        lines = []
        if config.api_env and config.api_env.strip():
            for env_line in config.api_env.strip().split("\n"):
                if env_line.strip():
                    lines.append(env_line)
            lines.append("")
        if config.activate_cmd and config.activate_cmd.strip():
            lines.append(config.activate_cmd)
            lines.append("")
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
        lines.append(" \\\n    ".join(parts))
        return "\n".join(lines)

    def _build_shell_command(self) -> str:
        config = self.app.config
        parts = ["python -m lmms_eval"]
        parts.append(f"--model {config.model}")
        if config.model_args:
            parts.append(f"--model_args '{config.model_args}'")
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
        cmd = " ".join(parts)
        if config.activate_cmd and config.activate_cmd.strip():
            cmd = f"{config.activate_cmd} && {cmd}"
        return cmd

    def _parse_env_vars(self) -> dict[str, str]:
        import re

        env = os.environ.copy()
        config = self.app.config
        if not config.api_env:
            return env
        for line in config.api_env.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:]
            match = re.match(r"(\w+)=(.*)", line)
            if match:
                key = match.group(1)
                value = match.group(2).strip()
                if (value.startswith('"') and value.endswith('"')) or (
                    value.startswith("'") and value.endswith("'")
                ):
                    value = value[1:-1]
                expanded = os.path.expandvars(value)
                env[key] = expanded
        return env

    @work(thread=True, exclusive=True)
    def _run_evaluation_worker(self) -> None:
        self._stop_requested = False
        cmd = self._build_shell_command()
        env = self._parse_env_vars()

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
                shell=True,
                start_new_session=True,
            )
            output_lines: list[str] = []
            if self._process.stdout:
                try:
                    for line in iter(self._process.stdout.readline, ""):
                        if self._stop_requested:
                            break
                        output_lines.append(_strip_ansi(line.rstrip()))
                        if len(output_lines) > 100:
                            output_lines = output_lines[-100:]
                        escaped_text = rich_escape("\n".join(output_lines))
                        output_text = _highlight_bash_output(escaped_text)
                        self.app.call_from_thread(self._update_output, output_text)
                except (ValueError, OSError):
                    pass
            if self._stop_requested:
                return
            if self._process:
                self._process.wait(timeout=5)
                returncode = self._process.returncode
                if returncode == 0:
                    self.app.call_from_thread(
                        self._update_status, "[bold green]Completed successfully![/]"
                    )
                else:
                    self.app.call_from_thread(
                        self._update_status,
                        f"[bold red]Failed with code {returncode}[/]",
                    )
        except subprocess.TimeoutExpired:
            if self._process:
                self._process.kill()
        except Exception as e:
            if not self._stop_requested:
                self.app.call_from_thread(
                    self._update_status, f"[bold red]Error: {rich_escape(str(e))}[/]"
                )
        finally:
            self._process = None
            self.app.call_from_thread(self._reset_buttons)

    def _update_output(self, text: str) -> None:
        self._has_output = True
        self._stop_loading_animation()
        try:
            self.query_one("#run-output", Static).update(text)
        except Exception:
            pass

    def _update_status(self, text: str) -> None:
        try:
            self.query_one("#run-status", Static).update(text)
        except Exception:
            pass

    def _reset_buttons(self) -> None:
        self._stop_loading_animation()
        try:
            self.query_one("#start-btn", Button).disabled = False
            self.query_one("#stop-btn", Button).disabled = True
        except Exception:
            pass

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
            with Vertical(id="header-left"):
                yield Static("LMMs-Eval Terminal UI", id="header-left-title")
                with Horizontal(id="header-left-content"):
                    with Container(id="logo-container"):
                        yield LogoWidget(id="header-image")
                        yield Static(
                            "[bold #5f87af]LMMS[/]\n[bold #87afd7]EVAL[/]",
                            id="header-logo",
                        )
                    with Vertical(id="header-info"):
                        yield Static("LMMs-Eval SYSTEM", classes="blink")
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
                output_lines.append(_strip_ansi(line.rstrip()))
                if len(output_lines) > 100:
                    output_lines = output_lines[-100:]
                escaped = rich_escape("\n".join(output_lines))
                output_widget.update(_highlight_bash_output(escaped))

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
            status_widget.update(f"[bold red]Error: {rich_escape(str(e))}[/]")

        self.query_one("#execute-btn", Button).disabled = False

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_cancel(self) -> None:
        self.app.pop_screen()


class LmmsEvalTUI(App):
    CSS_PATH = "app.tcss"

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
