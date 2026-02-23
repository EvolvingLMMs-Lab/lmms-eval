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
import urllib.error
import urllib.request
import uuid
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

import yaml
from fastapi import FastAPI, HTTPException, Query
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


class ExportYamlRequest(BaseModel):
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


class ExportYamlResponse(BaseModel):
    yaml_content: str


class ImportYamlRequest(BaseModel):
    yaml_content: str


class ImportYamlResponse(BaseModel):
    model: str = ""
    model_args: str = ""
    tasks: list[str] = []
    env_vars: str = ""
    batch_size: int = 1
    limit: int | None = None
    output_path: str = "./logs/"
    log_samples: bool = False
    verbosity: str = "INFO"
    device: str | None = None


class LogRunSummary(BaseModel):
    run_id: str
    model_name: str
    date: str
    tasks: list[str]
    metrics: dict[str, dict[str, Any]]
    total_evaluation_time_seconds: Any | None = None
    config: dict[str, Any]
    n_samples: dict[str, Any]


class LogSamplesResponse(BaseModel):
    samples: list[dict[str, Any]]
    total: int
    offset: int
    limit: int


def _resolve_logs_root(logs_path: str) -> Path:
    return Path(logs_path).expanduser().resolve()


def _ensure_path_within_base(base_path: Path, target_path: Path) -> None:
    try:
        target_path.relative_to(base_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Path escapes logs directory") from exc


def _resolve_run_results_path(logs_path: str, run_id: str) -> Path:
    logs_root = _resolve_logs_root(logs_path)
    if not logs_root.exists() or not logs_root.is_dir():
        raise HTTPException(status_code=404, detail="Logs path not found")

    decoded_run_id = Path(unquote(run_id))
    if decoded_run_id.is_absolute():
        raise HTTPException(status_code=400, detail="Invalid run_id")

    run_path = (logs_root / decoded_run_id).resolve()
    _ensure_path_within_base(logs_root, run_path)
    return run_path


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


def _env_vars_to_dict(env_vars: str) -> dict[str, str]:
    """Convert env_vars multi-line string to a dict for YAML export."""
    env_dict: dict[str, str] = {}
    for line in env_vars.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[7:]
        if "=" in stripped:
            key, _, value = stripped.partition("=")
            env_dict[key.strip()] = value.strip()
    return env_dict


def _dict_to_env_vars(env_dict: dict[str, str]) -> str:
    """Convert env dict from YAML to env_vars multi-line string for UI."""
    lines = []
    for key, value in env_dict.items():
        lines.append(f"export {key}={value}")
    return "\n".join(lines)


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


@app.post("/eval/export-yaml", response_model=ExportYamlResponse)
async def export_yaml(request: ExportYamlRequest) -> ExportYamlResponse:
    """Export current UI config as a YAML config file."""
    config: dict[str, Any] = {}

    env_dict = _env_vars_to_dict(request.env_vars)
    if env_dict:
        config["env"] = env_dict

    config["model"] = request.model
    if request.model_args:
        config["model_args"] = request.model_args
    if request.tasks:
        config["tasks"] = ",".join(request.tasks)
    config["batch_size"] = request.batch_size
    if request.limit is not None:
        config["limit"] = request.limit
    config["output_path"] = request.output_path
    if request.log_samples:
        config["log_samples"] = True
    config["verbosity"] = request.verbosity
    if request.device:
        config["device"] = request.device

    header = "# LMMs-Eval config exported from Web UI\n" "# Usage: python -m lmms_eval --config <this_file>.yaml\n" "# CLI args override YAML values.\n\n"
    yaml_content = header + yaml.dump(config, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return ExportYamlResponse(yaml_content=yaml_content)


@app.post("/eval/import-yaml", response_model=ImportYamlResponse)
async def import_yaml(request: ImportYamlRequest) -> ImportYamlResponse:
    """Import a YAML config file into UI config values."""
    try:
        config = yaml.safe_load(request.yaml_content)
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {e}")

    if not isinstance(config, dict):
        raise HTTPException(status_code=400, detail="YAML must be a dict (not a list or scalar)")

    env_dict = config.pop("env", {})
    env_vars = _dict_to_env_vars(env_dict) if env_dict else ""

    tasks_raw = config.get("tasks", "")
    if isinstance(tasks_raw, str):
        tasks = [t.strip() for t in tasks_raw.split(",") if t.strip()]
    elif isinstance(tasks_raw, list):
        tasks = tasks_raw
    else:
        tasks = []

    return ImportYamlResponse(
        model=config.get("model", ""),
        model_args=config.get("model_args", ""),
        tasks=tasks,
        env_vars=env_vars,
        batch_size=config.get("batch_size", 1),
        limit=config.get("limit"),
        output_path=config.get("output_path", "./logs/"),
        log_samples=config.get("log_samples", False),
        verbosity=config.get("verbosity", "INFO"),
        device=config.get("device"),
    )


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


@app.get("/logs/runs", response_model=list[LogRunSummary])
async def list_log_runs(logs_path: str = Query("./logs/")) -> list[LogRunSummary]:
    logs_root = _resolve_logs_root(logs_path)
    if not logs_root.exists() or not logs_root.is_dir():
        return []

    runs: list[LogRunSummary] = []

    for results_file in logs_root.rglob("*_results.json"):
        resolved_file = results_file.resolve()
        try:
            _ensure_path_within_base(logs_root, resolved_file)
        except HTTPException:
            continue

        try:
            with resolved_file.open("r", encoding="utf-8") as f:
                result_data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        if not isinstance(result_data, dict):
            continue

        task_results = result_data.get("results")
        if not isinstance(task_results, dict):
            task_results = {}

        metrics: dict[str, dict[str, Any]] = {}
        for task_name, task_metrics in task_results.items():
            if not isinstance(task_metrics, dict):
                continue
            metrics[str(task_name)] = {str(metric_name): metric_value for metric_name, metric_value in task_metrics.items() if metric_name != "alias"}

        config = result_data.get("config")
        if not isinstance(config, dict):
            config = {}

        n_samples = result_data.get("n-samples")
        if not isinstance(n_samples, dict):
            n_samples = {}

        date = result_data.get("date")
        if date is None:
            date = resolved_file.stem.removesuffix("_results")

        model_name = result_data.get("model_name")
        if model_name is None:
            model_name = ""

        relative_path = resolved_file.relative_to(logs_root).as_posix()

        runs.append(
            LogRunSummary(
                run_id=quote(relative_path, safe=""),
                model_name=str(model_name),
                date=str(date),
                tasks=[str(task_name) for task_name in task_results.keys()],
                metrics=metrics,
                total_evaluation_time_seconds=result_data.get("total_evaluation_time_seconds"),
                config=config,
                n_samples=n_samples,
            )
        )

    runs.sort(key=lambda run: run.date, reverse=True)
    return runs


@app.get("/logs/runs/{run_id:path}/results")
async def get_log_run_results(
    run_id: str,
    logs_path: str = Query("./logs/"),
) -> dict[str, Any]:
    run_path = _resolve_run_results_path(logs_path, run_id)
    if not run_path.name.endswith("_results.json"):
        raise HTTPException(status_code=404, detail="Run results not found")
    if not run_path.exists() or not run_path.is_file():
        raise HTTPException(status_code=404, detail="Run results not found")

    try:
        with run_path.open("r", encoding="utf-8") as f:
            result_data = json.load(f)
    except OSError as exc:
        raise HTTPException(status_code=404, detail="Run results not found") from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="Run results JSON is invalid") from exc

    if not isinstance(result_data, dict):
        raise HTTPException(status_code=500, detail="Run results must be a JSON object")

    return result_data


@app.get("/logs/runs/{run_id:path}/samples/{task_name}", response_model=LogSamplesResponse)
async def get_log_run_samples(
    run_id: str,
    task_name: str,
    logs_path: str = Query("./logs/"),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
) -> LogSamplesResponse:
    run_path = _resolve_run_results_path(logs_path, run_id)
    if not run_path.name.endswith("_results.json"):
        raise HTTPException(status_code=404, detail="Run results not found")
    if not run_path.exists() or not run_path.is_file():
        raise HTTPException(status_code=404, detail="Run results not found")
    if "/" in task_name or "\\" in task_name:
        raise HTTPException(status_code=400, detail="Invalid task name")

    run_stem = run_path.stem
    if not run_stem.endswith("_results"):
        raise HTTPException(status_code=404, detail="Run results not found")

    run_prefix = run_stem.removesuffix("_results")
    samples_path = run_path.with_name(f"{run_prefix}_samples_{task_name}.jsonl")

    if not samples_path.exists() or not samples_path.is_file():
        raise HTTPException(status_code=404, detail="Samples file not found")

    samples: list[dict[str, Any]] = []
    total = 0

    try:
        with samples_path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if not isinstance(sample, dict):
                    continue

                if total >= offset and len(samples) < limit:
                    samples.append(sample)
                total += 1
    except OSError as exc:
        raise HTTPException(status_code=404, detail="Samples file not found") from exc

    return LogSamplesResponse(samples=samples, total=total, offset=offset, limit=limit)


@app.get("/logs/dataset-row")
def get_dataset_row(
    dataset_path: str = Query(..., description="HuggingFace dataset path, e.g. lmms-lab/MME"),
    split: str = Query(..., description="Dataset split, e.g. test"),
    doc_id: int = Query(..., ge=0, description="Document index in the dataset"),
    config: str = Query("default", description="Dataset config name"),
) -> dict[str, Any]:
    """Fetch a single row from a HuggingFace dataset via the datasets server API.

    Returns the row data including image URLs that can be rendered directly by the frontend.
    Images appear as objects with a 'src' key pointing to a CDN URL.
    """
    api_url = f"https://datasets-server.huggingface.co/rows" f"?dataset={quote(dataset_path, safe='')}" f"&config={quote(config, safe='')}" f"&split={quote(split, safe='')}" f"&offset={doc_id}" f"&length=1"

    req = urllib.request.Request(api_url)
    req.add_header("Accept", "application/json")

    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        req.add_header("Authorization", f"Bearer {hf_token}")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise HTTPException(status_code=404, detail="Dataset or row not found on HuggingFace") from exc
        raise HTTPException(status_code=exc.code, detail=f"HuggingFace API error: {exc.reason}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise HTTPException(status_code=502, detail=f"Failed to reach HuggingFace API: {exc}") from exc

    rows = data.get("rows", [])
    if not rows:
        raise HTTPException(status_code=404, detail="Row not found in dataset")

    row = rows[0].get("row", {})
    return {"row": row}


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
