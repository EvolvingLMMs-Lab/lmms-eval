"""lmms-eval models — list registered model backends and aliases."""

from __future__ import annotations

import argparse


def add_models_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "models",
        help="List available model backends",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--aliases",
        action="store_true",
        default=False,
        help="Include model aliases in the listing",
    )
    p.set_defaults(func=run_models)


def _col(text: str, width: int) -> str:
    """Left-align text in a fixed-width column."""
    return text[:width].ljust(width)


def run_models(args: argparse.Namespace) -> None:
    from lmms_eval.models import (
        AVAILABLE_CHAT_TEMPLATE_MODELS,
        AVAILABLE_SIMPLE_MODELS,
        MODEL_ALIASES,
    )

    chat_only = sorted(set(AVAILABLE_CHAT_TEMPLATE_MODELS) - set(AVAILABLE_SIMPLE_MODELS))
    simple_only = sorted(set(AVAILABLE_SIMPLE_MODELS) - set(AVAILABLE_CHAT_TEMPLATE_MODELS))
    dual = sorted(set(AVAILABLE_CHAT_TEMPLATE_MODELS) & set(AVAILABLE_SIMPLE_MODELS))

    def _alias_str(name: str) -> str:
        aliases = MODEL_ALIASES.get(name, ())
        return ", ".join(aliases) if aliases else ""

    show_aliases = getattr(args, "aliases", False)

    if show_aliases:
        header = f"{_col('Name', 28)}{_col('Type', 14)}{_col('Aliases', 40)}"
        sep = "-" * 82
    else:
        header = f"{_col('Name', 28)}{_col('Type', 14)}"
        sep = "-" * 42
    print(f"\nRegistered Models ({len(chat_only) + len(simple_only) + len(dual)} total)\n")
    print(header)
    print(sep)

    def _print_row(name: str, typ: str) -> None:
        if show_aliases:
            alias = _alias_str(name)
            print(f"{_col(name, 28)}{_col(typ, 14)}{alias}")
        else:
            print(f"{_col(name, 28)}{_col(typ, 14)}")

    # Chat-only models first (recommended)
    for name in chat_only:
        _print_row(name, "chat")
    # Dual-mode models
    for name in dual:
        _print_row(name, "chat+simple")
    # Simple-only models
    for name in simple_only:
        _print_row(name, "simple")

    print(sep)
    print(f"\n  chat-only: {len(chat_only)}  |  chat+simple: {len(dual)}  |  simple-only: {len(simple_only)}")
    print("  Tip: chat models are recommended for new evaluations.\n")
