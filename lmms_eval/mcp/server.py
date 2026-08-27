from __future__ import annotations

from contextlib import asynccontextmanager

from loguru import logger
from mcp.server.fastmcp import FastMCP

from lmms_eval.entrypoints.job_scheduler import JobScheduler
from lmms_eval.mcp.tools import register_tools


@asynccontextmanager
async def _lifespan(server: FastMCP):
    """Manage JobScheduler lifecycle alongside the MCP server."""
    scheduler = JobScheduler()
    await scheduler.start()
    register_tools(server, scheduler)
    logger.info("[MCP] lmms-eval MCP server started with JobScheduler")
    try:
        yield
    finally:
        await scheduler.stop()
        logger.info("[MCP] lmms-eval MCP server stopped")


def create_mcp_server() -> FastMCP:
    """Create and configure the lmms-eval MCP server."""
    mcp = FastMCP(
        "lmms-eval",
        instructions=(
            "lmms-eval MCP Server - Evaluate large multimodal models.\n\n"
            "Available tools:\n"
            "- list_tasks / get_task_info: Discover evaluation benchmarks\n"
            "- list_models / get_model_info: Discover model backends\n"
            "- evaluate: Run evaluations (async with job tracking)\n"
            "- get_run_status / get_run_result: Monitor evaluation runs\n"
            "- cancel_run: Cancel queued evaluations"
        ),
        json_response=True,
        lifespan=_lifespan,
    )
    return mcp
