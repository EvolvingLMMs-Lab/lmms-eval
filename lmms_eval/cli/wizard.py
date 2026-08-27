"""Interactive evaluation wizard for `lmms-eval eval` (no-args mode).

Pure Python, no external dependencies.  Uses numbered menus and simple
input() prompts so it works in any terminal.
"""

from __future__ import annotations

import shlex
import sys

# ── helpers ──────────────────────────────────────────────────────────


def _ask(prompt: str, default: str = "") -> str:
    """Prompt the user, returning *default* on empty input."""
    suffix = f" [{default}]" if default else ""
    try:
        value = input(f"  {prompt}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        sys.exit(0)
    return value or default


def _ask_yn(prompt: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    val = _ask(prompt, hint).lower()
    if val in ("y/n", "y", "yes"):
        return True
    if val in ("n", "no"):
        return False
    return default


def _pick_from_list(
    items: list[str],
    header: str,
    *,
    page_size: int = 20,
    allow_custom: bool = False,
) -> str:
    """Show a numbered list and let the user pick one entry."""
    print(f"\n  {header} ({len(items)} available)")
    print()

    # Show first page
    show = items[:page_size]
    for i, name in enumerate(show, 1):
        print(f"    {i:3d}. {name}")
    if len(items) > page_size:
        print(f"    ... and {len(items) - page_size} more (type 'all' to show full list)")

    while True:
        raw = _ask("Select [number or name]").strip()
        if raw.lower() == "all":
            print()
            for i, name in enumerate(items, 1):
                print(f"    {i:3d}. {name}")
            continue
        # Try as number
        try:
            idx = int(raw)
            if 1 <= idx <= len(items):
                return items[idx - 1]
            print(f"    Number out of range (1-{len(items)})")
            continue
        except ValueError:
            pass
        # Try as name (exact or prefix match)
        exact = [n for n in items if n == raw]
        if exact:
            return exact[0]
        prefix = [n for n in items if n.startswith(raw)]
        if len(prefix) == 1:
            return prefix[0]
        if len(prefix) > 1:
            print(f"    Ambiguous: {', '.join(prefix[:5])}{'...' if len(prefix) > 5 else ''}")
            continue
        if allow_custom:
            return raw
        print(f"    Not found: {raw}")


# ── wizard steps ─────────────────────────────────────────────────────


def _step_model() -> tuple[str, str]:
    """Step 1: pick model and model_args."""
    from lmms_eval.models import (
        AVAILABLE_CHAT_TEMPLATE_MODELS,
        AVAILABLE_SIMPLE_MODELS,
    )

    # Build ordered list: chat first (recommended), then simple-only
    chat_names = sorted(AVAILABLE_CHAT_TEMPLATE_MODELS.keys())
    simple_only = sorted(set(AVAILABLE_SIMPLE_MODELS.keys()) - set(chat_names))

    # Label chat models with a marker
    labeled: list[tuple[str, str]] = []
    for n in chat_names:
        tag = "chat+simple" if n in AVAILABLE_SIMPLE_MODELS else "chat"
        labeled.append((n, tag))
    for n in simple_only:
        labeled.append((n, "simple"))

    print("\n" + "=" * 60)
    print("  Step 1/4 \u2014 Select model")
    print("=" * 60)
    print()
    print("  Chat models (recommended):")
    for i, (name, tag) in enumerate(labeled, 1):
        if tag.startswith("chat"):
            print(f"    {i:3d}. {name}")
        if i == 15 and len(labeled) > 20:
            remaining_chat = sum(1 for _, t in labeled[i:] if t.startswith("chat"))
            if remaining_chat > 0:
                print(f"         ... +{remaining_chat} more chat models")
            break

    print("\n  Simple models (legacy):")
    simple_start = len(chat_names) + 1
    shown = 0
    for i, (name, tag) in enumerate(labeled, 1):
        if tag == "simple":
            print(f"    {i:3d}. {name}")
            shown += 1
            if shown >= 10:
                remaining = len(simple_only) - shown
                if remaining > 0:
                    print(f"         ... +{remaining} more (type 'all')")
                break

    all_names = [n for n, _ in labeled]
    model_name = _pick_from_list(all_names, "Models", page_size=999, allow_custom=True)

    print(f"\n  Selected: {model_name}")

    # Model args
    print()
    model_args = _ask("model_args (e.g. pretrained=Qwen/Qwen2.5-VL-7B-Instruct)", "")

    return model_name, model_args


def _step_tasks() -> str:
    """Step 2: pick tasks."""
    from lmms_eval.tasks import TaskManager

    print("\n" + "=" * 60)
    print("  Step 2/4 \u2014 Select tasks")
    print("=" * 60)

    tm = TaskManager("WARNING")
    all_tasks = sorted(tm.all_subtasks)

    print("\n  Options:")
    print("    - Type task name(s) directly (comma-separated)")
    print("    - Type 'list' to show all tasks")
    print("    - Type 'search <keyword>' to filter\n")

    while True:
        raw = _ask("Tasks").strip()
        if not raw:
            print("    Please enter at least one task name.")
            continue
        if raw == "list":
            print()
            for i, t in enumerate(all_tasks, 1):
                print(f"    {i:4d}. {t}")
            continue
        if raw.startswith("search "):
            keyword = raw[7:].lower()
            matches = [t for t in all_tasks if keyword in t.lower()]
            if matches:
                print(f"\n  Matches ({len(matches)}):")
                for t in matches:
                    print(f"    {t}")
            else:
                print(f"    No tasks match '{keyword}'")
            continue
        # Validate task names
        names = [n.strip() for n in raw.split(",") if n.strip()]
        unknown = [n for n in names if n not in all_tasks]
        if unknown:
            print(f"    Unknown tasks: {', '.join(unknown)}")
            print("    Type 'list' to see all available tasks.")
            continue
        return ",".join(names)


def _step_options() -> dict[str, str]:
    """Step 3: common evaluation options."""
    print("\n" + "=" * 60)
    print("  Step 3/4 \u2014 Options")
    print("=" * 60)
    print()

    opts: dict[str, str] = {}

    batch = _ask("Batch size", "1")
    opts["batch_size"] = batch

    limit = _ask("Limit (samples per task, -1=all)", "8")
    if limit != "-1":
        opts["limit"] = limit

    device = _ask("Device", "cuda")
    opts["device"] = device

    log_samples = _ask_yn("Log samples?", default=True)
    if log_samples:
        opts["log_samples"] = "true"
        output = _ask("Output path", "./logs")
        opts["output_path"] = output

    return opts


def _build_command(model: str, model_args: str, tasks: str, opts: dict[str, str]) -> list[str]:
    """Build the full CLI command list."""
    cmd = ["lmms-eval", "eval", "--model", model]
    if model_args:
        cmd += ["--model_args", model_args]
    cmd += ["--tasks", tasks]
    for k, v in opts.items():
        flag = f"--{k}"
        if v == "true":
            cmd.append(flag)
        else:
            cmd += [flag, v]
    return cmd


# ── main entry ───────────────────────────────────────────────────────


def run_wizard() -> None:
    """Run the interactive evaluation wizard."""
    print()
    print("\u250c" + "\u2500" * 58 + "\u2510")
    print("\u2502  LMMs-Eval Interactive Wizard" + " " * 28 + "\u2502")
    print("\u2502  Configure and launch an evaluation step by step." + " " * 6 + "\u2502")
    print("\u2514" + "\u2500" * 58 + "\u2518")

    model, model_args = _step_model()
    tasks = _step_tasks()
    opts = _step_options()
    cmd = _build_command(model, model_args, tasks, opts)

    # Step 4: confirm
    print("\n" + "=" * 60)
    print("  Step 4/4 \u2014 Confirm")
    print("=" * 60)
    print()

    # Pretty-print the command
    cmd_str = cmd[0]
    for part in cmd[1:]:
        if part.startswith("--"):
            cmd_str += f" \\\n    {part}"
        else:
            cmd_str += f" {shlex.quote(part)}" if " " in part else f" {part}"
    print(f"  {cmd_str}")
    print()

    if not _ask_yn("Run this evaluation?", default=True):
        print("\n  Command not executed. You can copy the above and run it manually.")
        return

    # Execute
    print("\n" + "-" * 60)
    print("  Starting evaluation...")
    print("-" * 60 + "\n")
    # Set sys.argv so that cli_evaluate's parse_eval_args() sees the
    # correct flags.  cli_evaluate() always re-parses from sys.argv,
    # so passing a Namespace alone is not enough.
    sys.argv = cmd

    from lmms_eval.__main__ import cli_evaluate

    cli_evaluate()
