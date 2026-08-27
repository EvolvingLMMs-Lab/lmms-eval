"""lmms-eval ui — launch the web-based evaluation UI."""

from __future__ import annotations

import argparse


def add_ui_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "ui",
        help="Launch the Web UI for interactive evaluation",
    )
    p.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to serve on (default: 8000)",
    )
    p.set_defaults(func=run_ui)


def run_ui(args: argparse.Namespace) -> None:
    import os

    # Honour env var, but CLI flag takes precedence
    if args.port != 8000:
        os.environ["LMMS_SERVER_PORT"] = str(args.port)
    elif "LMMS_SERVER_PORT" not in os.environ:
        os.environ["LMMS_SERVER_PORT"] = "8000"

    from lmms_eval.tui.cli import main as tui_web_main

    raise SystemExit(tui_web_main())
