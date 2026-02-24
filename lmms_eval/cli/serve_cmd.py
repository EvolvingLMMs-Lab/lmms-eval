"""lmms-eval serve — start the HTTP evaluation server."""

from __future__ import annotations

import argparse


def add_serve_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "serve",
        help="Start the HTTP evaluation server for async/remote evaluations",
    )
    p.add_argument("--host", type=str, default="localhost", help="Host to bind to (default: localhost)")
    p.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    p.set_defaults(func=run_serve)


def run_serve(args: argparse.Namespace) -> None:
    from lmms_eval.entrypoints import ServerArgs, launch_server

    server_args = ServerArgs(host=args.host, port=args.port)
    try:
        launch_server(server_args)
    except KeyboardInterrupt:
        print("\nServer shutdown by user")
