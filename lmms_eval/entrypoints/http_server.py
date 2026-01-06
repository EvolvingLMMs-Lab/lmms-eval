import asyncio
import json
import tempfile
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from lmms_eval.entrypoints.server_args import ServerArgs

# =============================================================================
# Enums and Models
# =============================================================================


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EvaluateRequest(BaseModel):
    """Request model for evaluation endpoint."""

    model: str = Field(..., description="Model name or path")
    tasks: List[str] = Field(..., description="List of task names to evaluate")
    model_args: Optional[Dict[str, Any]] = Field(default=None, description="Model arguments")
    launcher_args: Optional[Dict[str, Any]] = Field(default=None, description="Launcher arguments")
    num_fewshot: Optional[int] = Field(default=None, description="Number of few-shot examples")
    batch_size: Optional[Union[int, str]] = Field(default=None, description="Batch size")
    device: Optional[str] = Field(default=None, description="Device to run on")
    limit: Optional[Union[int, float]] = Field(default=None, description="Limit number of examples")
    gen_kwargs: Optional[str] = Field(default=None, description="Generation kwargs")
    log_samples: bool = Field(default=True, description="Whether to log samples")
    predict_only: bool = Field(default=False, description="Only generate predictions")
    num_gpus: int = Field(default=1, description="Number of GPUs to use")
    output_dir: Optional[str] = Field(default=None, description="Output directory for results")


class JobInfo(BaseModel):
    """Information about a job."""

    job_id: str
    status: JobStatus
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    request: EvaluateRequest
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    position_in_queue: Optional[int] = None


class JobSubmitResponse(BaseModel):
    """Response when submitting a job."""

    job_id: str
    status: JobStatus
    position_in_queue: int
    message: str


class QueueStatusResponse(BaseModel):
    """Response for queue status."""

    queue_size: int
    running_job: Optional[str] = None
    queued_jobs: List[str]
    completed_jobs: int
    failed_jobs: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: str
    queue_size: int


# =============================================================================
# Global State
# =============================================================================


class ServerState:
    """Global server state container."""

    def __init__(self):
        self.job_queue: asyncio.Queue = None
        self.jobs: Dict[str, JobInfo] = {}
        self.worker_task: asyncio.Task = None
        self.current_job_id: Optional[str] = None

    def reset(self):
        self.job_queue = asyncio.Queue()
        self.jobs = {}
        self.worker_task = None
        self.current_job_id = None


state = ServerState()


# =============================================================================
# Job Execution
# =============================================================================


def parse_output_directory(output_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse output directory: output_path/model_name/YYYYMMDD_HHMMSS_results.json

    Returns:
        {model_name: {"results": path, "samples": [paths]}}
    """
    output_dir = Path(output_path)
    if not output_dir.exists():
        return {}

    result = {}

    for model_dir in output_dir.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name

        # Group files by timestamp
        timestamps = defaultdict(lambda: {"results": None, "samples": []})

        for file in model_dir.glob("*_results.json"):
            timestamp = file.name.split("_results.json")[0]
            timestamps[timestamp]["results"] = str(file)

        for file in model_dir.glob("*_samples_*.jsonl"):
            # Extract timestamp (everything before first _samples_)
            timestamp = file.name.split("_samples_")[0]
            timestamps[timestamp]["samples"].append(str(file))

        if not timestamps:
            continue

        # Use latest timestamp
        sorted_ts = sorted(timestamps.keys(), reverse=True)
        if len(sorted_ts) > 1:
            print(f"[WARNING] Multiple timestamps for '{model_name}': {sorted_ts}. Using latest.")

        result[model_name] = timestamps[sorted_ts[0]]

    return result


async def run_evaluation_subprocess(config: dict) -> dict:
    """
    Run evaluation in a subprocess using accelerate/torchrun.

    This allows GPU-based evaluation to run in a separate process
    while the server remains responsive.
    """
    # Create temporary files for config and output
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        config_path = f.name

    output_path = config.get("output_dir", tempfile.mktemp(suffix="_results"))

    # Build command
    num_gpus = config.get("num_gpus", 1)

    # Use accelerate launch for multi-GPU support
    cmd = [
        "accelerate",
        "launch",
        "--num_processes",
        str(num_gpus),
        "-m",
        "lmms_eval",
        "--model",
        config["model"],
        "--tasks",
        ",".join(config["tasks"]),
        "--output_path",
        output_path,
    ]

    # Add optional arguments
    if config.get("model_args"):
        if isinstance(config["model_args"], dict):
            model_args_str = ",".join(f"{k}={v}" for k, v in config["model_args"].items())
        else:
            model_args_str = config["model_args"]
        cmd.extend(["--model_args", model_args_str])

    if config.get("batch_size"):
        cmd.extend(["--batch_size", str(config["batch_size"])])

    if config.get("limit"):
        cmd.extend(["--limit", str(config["limit"])])

    if config.get("num_fewshot") is not None:
        cmd.extend(["--num_fewshot", str(config["num_fewshot"])])

    if config.get("gen_kwargs"):
        cmd.extend(["--gen_kwargs", config["gen_kwargs"]])

    if config.get("log_samples"):
        cmd.append("--log_samples")

    if config.get("predict_only"):
        cmd.append("--predict_only")

    # Run subprocess with streaming output
    print(f"[EVAL] Launching: {' '.join(cmd)}")

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,  # Combine stderr into stdout
    )

    # Stream output in real-time
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        print(f"[EVAL] {line.decode().rstrip()}")

    await proc.wait()

    # Clean up config file
    Path(config_path).unlink(missing_ok=True)

    if proc.returncode != 0:
        raise RuntimeError(f"Evaluation failed with return code {proc.returncode}")

    # Parse and return results from output directory
    return parse_output_directory(output_path)


async def job_worker():
    """
    Background worker that processes jobs one-by-one from the queue.

    This ensures sequential execution of GPU-intensive evaluation jobs.
    """
    while True:
        try:
            job_id = await state.job_queue.get()
            job = state.jobs.get(job_id)

            if job is None:
                state.job_queue.task_done()
                continue

            # Update job status
            state.current_job_id = job_id
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now().isoformat()

            try:
                # Run evaluation
                config = job.request.model_dump()
                result = await run_evaluation_subprocess(config)

                # Update job with results
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.now().isoformat()
                job.result = result

            except Exception as e:
                job.status = JobStatus.FAILED
                job.completed_at = datetime.now().isoformat()
                job.error = str(e)

            finally:
                state.current_job_id = None
                state.job_queue.task_done()

        except asyncio.CancelledError:
            break
        except Exception as e:
            # Log error but keep worker running
            print(f"Worker error: {e}")


# =============================================================================
# FastAPI App
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage server lifecycle - start/stop background worker."""
    # Startup
    state.reset()
    state.job_queue = asyncio.Queue()
    state.worker_task = asyncio.create_task(job_worker())
    print("Evaluation server started, worker ready")

    yield

    # Shutdown
    if state.worker_task:
        state.worker_task.cancel()
        try:
            await state.worker_task
        except asyncio.CancelledError:
            pass
    print("Evaluation server shutdown complete")


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
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        queue_size=state.job_queue.qsize() if state.job_queue else 0,
    )


@app.post("/evaluate", response_model=JobSubmitResponse)
async def submit_evaluation(request: EvaluateRequest):
    """
    Submit an evaluation job to the queue.

    The job will be queued and processed sequentially.
    Use GET /jobs/{job_id} to check status and retrieve results.
    """
    job_id = str(uuid.uuid4())
    position = state.job_queue.qsize()

    # Create job info
    job = JobInfo(
        job_id=job_id,
        status=JobStatus.QUEUED,
        created_at=datetime.now().isoformat(),
        request=request,
        position_in_queue=position,
    )
    state.jobs[job_id] = job

    # Add to queue
    await state.job_queue.put(job_id)

    return JobSubmitResponse(
        job_id=job_id,
        status=JobStatus.QUEUED,
        position_in_queue=position,
        message=f"Job queued successfully. Position in queue: {position}",
    )


@app.get("/jobs/{job_id}", response_model=JobInfo)
async def get_job_status(job_id: str):
    """Get the status and results of a job."""
    job = state.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Update position in queue if still queued
    if job.status == JobStatus.QUEUED:
        # Count jobs ahead in queue
        position = 0
        for jid, j in state.jobs.items():
            if j.status == JobStatus.QUEUED and j.created_at < job.created_at:
                position += 1
        job.position_in_queue = position

    return job


@app.get("/queue", response_model=QueueStatusResponse)
async def get_queue_status():
    """Get the current queue status."""
    queued = [jid for jid, j in state.jobs.items() if j.status == JobStatus.QUEUED]
    completed = sum(1 for j in state.jobs.values() if j.status == JobStatus.COMPLETED)
    failed = sum(1 for j in state.jobs.values() if j.status == JobStatus.FAILED)

    return QueueStatusResponse(
        queue_size=len(queued),
        running_job=state.current_job_id,
        queued_jobs=queued,
        completed_jobs=completed,
        failed_jobs=failed,
    )


@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """
    Cancel a queued job.

    Note: Running jobs cannot be cancelled.
    """
    job = state.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job.status == JobStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Cannot cancel a running job")

    if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
        raise HTTPException(status_code=400, detail="Job already finished")

    # Remove from jobs dict (it will be skipped by worker)
    del state.jobs[job_id]

    return {"message": f"Job {job_id} cancelled"}


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
        args: ServerArgs instance containing host and port configuration.

    Example:
        >>> from lmms_eval.entrypoints import ServerArgs, launch_server
        >>> args = ServerArgs(host="0.0.0.0", port=8080)
        >>> launch_server(args)
    """
    print(f"Starting LMMS-Eval server at http://{args.host}:{args.port}")
    print("API docs available at /docs")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )
