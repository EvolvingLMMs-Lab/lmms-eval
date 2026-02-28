"""lmms-eval mcp - start the MCP evaluation server."""

from __future__ import annotations

import argparse


def add_mcp_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "mcp",
        help="Start the MCP (Model Context Protocol) server for AI agent integration",
    )
    p.add_argument("--transport", type=str, default="stdio", choices=["stdio", "sse"], help="MCP transport type (default: stdio)")
    p.set_defaults(func=run_mcp)


def run_mcp(args: argparse.Namespace) -> None:
    try:
        from mcp.server.fastmcp import FastMCP  # noqa: F401
    except ImportError:
        print("MCP server requires the 'mcp' package. Install with: pip install 'lmms_eval[mcp]'")
        import sys

        sys.exit(1)

    from lmms_eval.mcp.server import create_mcp_server

    server = create_mcp_server()
    server.run(transport=args.transport)
