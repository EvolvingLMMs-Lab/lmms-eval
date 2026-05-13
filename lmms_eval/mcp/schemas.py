from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TaskInfo(BaseModel):
    """Information about a single evaluation task."""

    name: str
    type: str  # "task", "group", "tag"
    yaml_path: Optional[str] = None
    output_type: Optional[str] = None  # "generate_until", "loglikelihood", etc.


class TaskListResponse(BaseModel):
    """Response for list_tasks tool."""

    tasks: List[TaskInfo]
    total: int
    query: Optional[str] = None


class ModelInfo(BaseModel):
    """Information about a single model backend."""

    model_id: str
    has_chat: bool
    has_simple: bool
    aliases: List[str] = Field(default_factory=list)


class ModelListResponse(BaseModel):
    """Response for list_models tool."""

    models: List[ModelInfo]
    total: int


class EvalRunSubmitted(BaseModel):
    """Response when an evaluation run is submitted."""

    run_id: str
    status: str  # "running" or "queued"
    position_in_queue: Optional[int] = None
    message: str


class EvalRunStatus(BaseModel):
    """Response for get_run_status tool."""

    run_id: str
    status: str  # "queued", "running", "completed", "failed", "cancelled"
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    position_in_queue: Optional[int] = None
    error: Optional[str] = None


class EvalRunResult(BaseModel):
    """Response for get_run_result tool."""

    run_id: str
    status: str
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
