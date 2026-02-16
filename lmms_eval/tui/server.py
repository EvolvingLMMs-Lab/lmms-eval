"""
LMMs-Eval Web UI Server - FastAPI backend with static file serving.
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import signal
import socket
import subprocess
import uuid
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from lmms_eval.tui.discovery import get_discovery_cache

app = FastAPI(title="LMMs-Eval Web UI", version="0.1.0")

STATIC_DIR = Path(__file__).parent / "web" / "dist"

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job storage
_jobs: dict[str, dict[str, Any]] = {}


def get_version() -> str:
    """Get lmms-eval version from package metadata."""
    try:
        return pkg_version("lmms_eval")
    except Exception:
        return "0.6.0"


def get_git_info() -> dict[str, str]:
    try:
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
        return {"branch": branch, "commit": commit}
    except Exception:
        return {"branch": "unknown", "commit": "unknown"}


def get_repo_root() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip()
    except Exception:
        return ""


def get_system_info() -> dict[str, str]:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cwd": os.getcwd(),
        "repo_root": get_repo_root(),
    }


# --- Models ---


class ModelInfo(BaseModel):
    id: str
    name: str


class TaskInfo(BaseModel):
    id: str
    name: str
    group: bool = False


class EvalRequest(BaseModel):
    model: str
    model_args: str = ""
    tasks: list[str]
    env_vars: str = ""
    batch_size: int = 1
    limit: int | None = 10
    output_path: str = "./logs/"
    log_samples: bool = True
    verbosity: str = "INFO"
    device: str | None = None


class EvalStartResponse(BaseModel):
    job_id: str
    command: str


class PreviewRequest(BaseModel):
    model: str
    model_args: str = ""
    tasks: list[str]
    env_vars: str = ""
    batch_size: int = 1
    limit: int | None = 10
    output_path: str = "./logs/"
    log_samples: bool = True
    verbosity: str = "INFO"
    device: str | None = None


class PreviewResponse(BaseModel):
    command: str


# --- Endpoints ---


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "version": get_version(),
        "git": get_git_info(),
        "system": get_system_info(),
    }


@app.get("/models", response_model=list[ModelInfo])
async def get_models() -> list[ModelInfo]:
    """Get available models."""
    cache = get_discovery_cache()
    models = cache.get_models(include_all=True)
    return [ModelInfo(id=model_id, name=name) for model_id, name in models]


@app.get("/tasks", response_model=list[TaskInfo])
async def get_tasks() -> list[TaskInfo]:
    """Get available tasks."""
    cache = get_discovery_cache()
    tasks = cache.get_tasks(include_all=True)
    return [
        TaskInfo(
            id=task_id,
            name=name,
            group=name.startswith("[Group]"),
        )
        for task_id, name in tasks
    ]


def _normalize_env_line(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        return stripped
    if "=" in stripped:
        return f"export {stripped}"
    return None


def _build_env_exports(env_vars: str) -> list[str]:
    exports: list[str] = []
    for line in env_vars.splitlines():
        export_line = _normalize_env_line(line)
        if export_line:
            exports.append(export_line)
    return exports


def _build_command(request: EvalRequest | PreviewRequest) -> str:
    """Build the lmms_eval command string."""
    parts = ["python -m lmms_eval"]
    parts.append(f"--model {request.model}")
    if request.model_args:
        parts.append(f"--model_args '{request.model_args}'")
    if request.tasks:
        parts.append(f"--tasks {','.join(request.tasks)}")
    parts.append(f"--batch_size {request.batch_size}")
    if request.limit is not None:
        parts.append(f"--limit {request.limit}")
    parts.append(f"--output_path {request.output_path}")
    if request.log_samples:
        parts.append("--log_samples")
    parts.append(f"--verbosity {request.verbosity}")
    if request.device:
        parts.append(f"--device {request.device}")
    command = " \\\n    ".join(parts)
    env_exports = _build_env_exports(request.env_vars)
    if env_exports:
        return "\n".join([*env_exports, command])
    return command


def _build_shell_command(request: EvalRequest) -> str:
    """Build the shell command for execution."""
    parts = ["python", "-m", "lmms_eval"]
    parts.extend(["--model", request.model])
    if request.model_args:
        parts.extend(["--model_args", request.model_args])
    if request.tasks:
        parts.extend(["--tasks", ",".join(request.tasks)])
    parts.extend(["--batch_size", str(request.batch_size)])
    if request.limit is not None:
        parts.extend(["--limit", str(request.limit)])
    parts.extend(["--output_path", request.output_path])
    if request.log_samples:
        parts.append("--log_samples")
    parts.extend(["--verbosity", request.verbosity])
    if request.device:
        parts.extend(["--device", request.device])
    command = " ".join(parts)
    env_exports = _build_env_exports(request.env_vars)
    if env_exports:
        export_prefix = " && ".join(env_exports)
        return f"{export_prefix} && {command}"
    return command


@app.post("/eval/preview", response_model=PreviewResponse)
async def preview_command(request: PreviewRequest) -> PreviewResponse:
    """Generate command preview without executing."""
    command = _build_command(request)
    return PreviewResponse(command=command)


@app.post("/eval/start", response_model=EvalStartResponse)
async def start_eval(request: EvalRequest) -> EvalStartResponse:
    """Start an evaluation job."""
    if not request.tasks:
        raise HTTPException(status_code=400, detail="No tasks specified")

    job_id = str(uuid.uuid4())
    command = _build_command(request)
    shell_command = _build_shell_command(request)

    _jobs[job_id] = {
        "status": "starting",
        "command": shell_command,
        "process": None,
        "request": request,
    }

    return EvalStartResponse(job_id=job_id, command=command)


async def _stream_output(job_id: str):
    """Stream subprocess output as SSE events."""
    job = _jobs.get(job_id)
    if not job:
        yield f"data: {json.dumps({'type': 'error', 'message': 'Job not found'})}\n\n"
        return

    shell_command = job["command"]

    try:
        process = await asyncio.create_subprocess_shell(
            shell_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        job["process"] = process
        job["status"] = "running"

        if process.stdout:
            async for line in process.stdout:
                if job.get("stopped"):
                    break
                decoded = line.decode("utf-8", errors="replace").rstrip()
                yield f"data: {json.dumps({'type': 'output', 'line': decoded})}\n\n"

        await process.wait()
        exit_code = process.returncode

        if job.get("stopped"):
            yield f"data: {json.dumps({'type': 'stopped'})}\n\n"
        else:
            yield f"data: {json.dumps({'type': 'done', 'exit_code': exit_code})}\n\n"

        job["status"] = "completed"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        job["status"] = "error"

    finally:
        job["process"] = None


@app.get("/eval/{job_id}/stream")
async def stream_eval(job_id: str):
    """Stream evaluation output via SSE."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return StreamingResponse(
        _stream_output(job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/eval/{job_id}/stop")
async def stop_eval(job_id: str) -> dict[str, str]:
    """Stop a running evaluation job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    job["stopped"] = True
    process = job.get("process")

    if process:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except (ProcessLookupError, OSError):
            try:
                process.terminate()
            except Exception:
                pass

    return {"status": "stopped"}


if STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    @app.get("/")
    async def serve_index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/{path:path}")
    async def serve_spa(path: str):
        file_path = STATIC_DIR / path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(STATIC_DIR / "index.html")
