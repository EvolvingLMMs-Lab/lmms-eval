"""Unified CLI dispatch for lmms-eval.

Routes to subcommands (tasks, models, eval, ui, serve, power, version)
while maintaining full backward compatibility with the flat-args form:

    lmms-eval --model X --tasks Y   # still works, routes to eval
"""

from __future__ import annotations

import argparse
import sys

# These are the known subcommand names.  If argv[1] is NOT one of these
# and starts with '-', we assume the legacy flat-args invocation and
# route everything to `eval`.
_SUBCOMMANDS = {"eval", "tasks", "models", "ui", "serve", "power", "version", "tui", "mcp"}

# Help banner shown when no args are given (before heavy imports).
_BANNER = """
\u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510
\u2502 LMMs-Eval: Evaluation framework for Large Multimodal Models  \u2502
\u251c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2524
\u2502                                                              \u2502
\u2502  lmms-eval eval   [--model X --tasks Y]   Run evaluation      \u2502
\u2502  lmms-eval tasks  [list|groups|subtasks]  Browse benchmarks   \u2502
\u2502  lmms-eval models [--aliases]             List model backends \u2502
\u2502  lmms-eval ui     [--port 8000]           Launch Web UI       \u2502
\u2502  lmms-eval serve  [--host H --port P]     HTTP eval server    \u2502
│  lmms-eval mcp    [--transport stdio]     MCP server (agents) │
\u2502  lmms-eval power  [--effect-size 0.03]    Power analysis      \u2502
\u2502  lmms-eval version                        Version info        \u2502
\u2502                                                              \u2502
\u2502  lmms-eval eval                           Interactive wizard  \u2502
\u2502  lmms-eval <subcommand> --help             Subcommand help    \u2502
\u2502                                                              \u2502
\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518
"""


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="lmms-eval",
        description="LMMs-Eval: Unified evaluation for Large Multimodal Models",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    sub = parser.add_subparsers(dest="subcommand")

    # ---- lightweight subcommands (no torch import) ----
    from lmms_eval.cli.mcp_cmd import add_mcp_parser
    from lmms_eval.cli.models_cmd import add_models_parser
    from lmms_eval.cli.power_cmd import add_power_parser
    from lmms_eval.cli.serve_cmd import add_serve_parser
    from lmms_eval.cli.tasks_cmd import add_tasks_parser
    from lmms_eval.cli.ui_cmd import add_ui_parser
    from lmms_eval.cli.version_cmd import add_version_parser

    add_tasks_parser(sub)
    add_models_parser(sub)
    add_ui_parser(sub)
    add_serve_parser(sub)
    add_power_parser(sub)
    add_version_parser(sub)
    add_mcp_parser(sub)

    # ---- eval subcommand (placeholder, actual parsing done by legacy code) ----
    eval_p = sub.add_parser(
        "eval",
        help="Run model evaluation (or launch interactive wizard with no args)",
        add_help=False,  # eval has its own rich --help
    )
    eval_p.set_defaults(func=None)  # handled specially below

    # ---- tui (textual TUI) ----
    tui_p = sub.add_parser("tui", help="Launch the terminal UI (requires textual)")
    tui_p.set_defaults(func=_run_tui)

    return parser


def _run_tui(_args: argparse.Namespace) -> None:
    try:
        from lmms_eval.tui.cli import main as tui_main

        tui_main()
    except ImportError as e:
        print("TUI mode requires 'textual' package. Install with: pip install lmms_eval[tui]")
        print(f"Error: {e}")
        sys.exit(1)


def _is_legacy_invocation(argv: list[str]) -> bool:
    """Detect the old flat-args form: `lmms-eval --model X --tasks Y`."""
    if not argv:
        return False
    first = argv[0]
    # If the first arg starts with '-', it's a flag (legacy flat-args form)
    return first.startswith("-")


def _is_eval_wizard(argv: list[str]) -> bool:
    """Detect `lmms-eval eval` with no further args (wizard mode)."""
    # `lmms-eval eval` with nothing else, or `lmms-eval eval --help`
    if len(argv) == 1 and argv[0] == "eval":
        return True
    return False


def main() -> None:
    """Unified CLI entry point."""
    argv = sys.argv[1:]

    # No args at all -> show banner
    if not argv:
        print(_BANNER)
        sys.exit(0)

    # --help / -h at top level
    if argv == ["--help"] or argv == ["-h"]:
        parser = _build_parser()
        parser.print_help()
        sys.exit(0)

    # Legacy --tui flag (handle before heavy imports)
    if "--tui" in argv:
        _run_tui(argparse.Namespace())
        sys.exit(0)
    if _is_eval_wizard(argv):
        from lmms_eval.cli.wizard import run_wizard

        run_wizard()
        sys.exit(0)

    # Legacy flat-args: `lmms-eval --model X --tasks Y ...`
    # Also covers: `lmms-eval eval --model X --tasks Y ...`
    if _is_legacy_invocation(argv) or (argv and argv[0] == "eval"):
        # Strip the 'eval' prefix if present so the legacy parser sees clean args
        if argv and argv[0] == "eval":
            argv = argv[1:]
        # Delegate to the original __main__.cli_evaluate with raw sys.argv
        sys.argv = [sys.argv[0]] + argv
        from lmms_eval.__main__ import cli_evaluate

        cli_evaluate()
        return

    # Subcommand dispatch
    parser = _build_parser()
    args = parser.parse_args(argv)

    if hasattr(args, "func") and args.func is not None:
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)
