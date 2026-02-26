"""lmms-eval MCP Server - Model Context Protocol interface for evaluation."""

from __future__ import annotations


def main() -> None:
    """Entry point for the lmms-eval MCP server (stdio transport)."""
    try:
        from mcp.server.fastmcp import FastMCP  # noqa: F401
    except ImportError:
        raise SystemExit("MCP server requires the 'mcp' package. Install with: pip install 'lmms_eval[mcp]'")

    from lmms_eval.mcp.server import create_mcp_server

    server = create_mcp_server()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
