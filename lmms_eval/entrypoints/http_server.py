"""
HTTP Server for LMMS-Eval.

This module provides a FastAPI-based HTTP server for running lmms-eval evaluations.
Jobs are managed through a JobScheduler that processes requests sequentially.
"""

from contextlib import asynccontextmanager
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from loguru import logger

from lmms_eval.entrypoints.job_scheduler import (
    EvaluateRequest,
    HealthResponse,
    JobInfo,
    JobScheduler,
    JobStatus,
    JobSubmitResponse,
    QueueStatusResponse,
)
from lmms_eval.entrypoints.server_args import ServerArgs

# =============================================================================
# Security Warning
# =============================================================================
# WARNING: This server is intended for use in trusted environments only.
# Do NOT expose this server to untrusted networks without additional security
# layers such as authentication, rate limiting, and network isolation.
# =============================================================================


# =============================================================================
# FastAPI App with Lifespan
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage server lifecycle.

    Initializes the JobScheduler and attaches it to app.state for
    access from route handlers.
    """
    # Startup: Initialize and start the job scheduler
    # Get scheduler config from app.state if available (set by launch_server)
    server_args: ServerArgs = getattr(app.state, "server_args", None)
    if server_args:
        scheduler = JobScheduler(
            max_completed_jobs=server_args.max_completed_jobs,
            temp_dir_prefix=server_args.temp_dir_prefix,
        )
    else:
        scheduler = JobScheduler()
    await scheduler.start()
    app.state.scheduler = scheduler
    logger.info("Evaluation server started")

    yield

    # Shutdown: Stop the scheduler
    await app.state.scheduler.stop()
    logger.info("Evaluation server shutdown complete")


app = FastAPI(
    title="LMMS-Eval Server",
    description="HTTP server for running lmms-eval evaluations",
    version="0.1.0",
    lifespan=lifespan,
)


# =============================================================================
# Routes
# =============================================================================


@app.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Health check endpoint."""
    scheduler: JobScheduler = request.app.state.scheduler
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        queue_size=scheduler.queue_size,
    )


@app.post("/evaluate", response_model=JobSubmitResponse)
async def submit_evaluation(request: Request, eval_request: EvaluateRequest):
    """
    Submit an evaluation job to the queue.

    The job will be queued and processed sequentially.
    Use GET /jobs/{job_id} to check status and retrieve results.
    """
    scheduler: JobScheduler = request.app.state.scheduler
    job_id, position = await scheduler.add_job(eval_request)

    return JobSubmitResponse(
        job_id=job_id,
        status=JobStatus.QUEUED,
        position_in_queue=position,
        message=f"Job queued successfully. Position in queue: {position}",
    )


@app.get("/jobs/{job_id}", response_model=JobInfo)
async def get_job_status(request: Request, job_id: str):
    """Get the status and results of a job."""
    scheduler: JobScheduler = request.app.state.scheduler
    job = await scheduler.get_job_with_position(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job


@app.get("/queue", response_model=QueueStatusResponse)
async def get_queue_status(request: Request):
    """Get the current queue status."""
    scheduler: JobScheduler = request.app.state.scheduler
    stats = await scheduler.get_queue_stats()

    return QueueStatusResponse(
        queue_size=len(stats["queued"]),
        running_job=stats["running_job"],
        queued_jobs=stats["queued"],
        completed_jobs=stats["completed"],
        failed_jobs=stats["failed"],
    )


@app.delete("/jobs/{job_id}")
async def cancel_job(request: Request, job_id: str):
    """
    Cancel a queued job.

    Note: Running jobs cannot be cancelled.
    """
    scheduler: JobScheduler = request.app.state.scheduler
    success, message = await scheduler.cancel_job(job_id)

    if not success:
        # Determine appropriate status code based on message
        if "not found" in message:
            raise HTTPException(status_code=404, detail=message)
        raise HTTPException(status_code=400, detail=message)

    return {"message": message}


@app.get("/tasks")
async def list_available_tasks():
    """List all available evaluation tasks."""
    try:
        from lmms_eval.tasks import TaskManager

        task_manager = TaskManager()
        return {"tasks": task_manager.all_tasks}
    except Exception as e:
        return {"tasks": [], "error": str(e)}


@app.get("/models")
async def list_available_models():
    """List all available model types."""
    try:
        from lmms_eval.models import AVAILABLE_MODELS

        return {"models": list(AVAILABLE_MODELS.keys())}
    except Exception as e:
        return {"models": [], "error": str(e)}


# =============================================================================
# Server Launch
# =============================================================================


def launch_server(args: ServerArgs):
    """
    Launch the evaluation server with the given arguments.

    Args:
        args: ServerArgs instance containing host, port, and scheduler configuration.

    Example:
        >>> from lmms_eval.entrypoints import ServerArgs, launch_server
        >>> args = ServerArgs(host="0.0.0.0", port=8080, max_completed_jobs=200)
        >>> launch_server(args)
    """
    logger.info(f"Starting LMMS-Eval server at http://{args.host}:{args.port}")
    logger.info("API docs available at /docs")

    # Store server args in app.state for the lifespan function to access
    app.state.server_args = args

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )
