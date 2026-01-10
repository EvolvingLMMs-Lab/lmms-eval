"""
API Protocol definitions for LMMS-Eval Server.

This module contains all Pydantic models used for request/response
validation across the HTTP server and client.
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Status of an evaluation job."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EvaluateRequest(BaseModel):
    """Request model for evaluation endpoint."""

    model: str = Field(..., description="Model name or path")
    tasks: List[str] = Field(..., description="List of task names to evaluate")
    model_args: Optional[Dict[str, Any]] = Field(default=None, description="Model arguments")
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


class MergeRequest(BaseModel):
    """Request model for merging FSDP2 sharded checkpoints."""

    checkpoint_path: str = Field(..., description="Path to sharded checkpoint directory")
    output_path: Optional[str] = Field(default=None, description="Output path for merged checkpoint")
    checkpoint_type: Literal["regular", "ema"] = Field(
        default="regular",
        description="Type of checkpoint to merge: 'regular' for main model weights, 'ema' for EMA weights",
    )


class MergeResponse(BaseModel):
    """Response model for checkpoint merge operations."""

    success: bool = Field(..., description="Whether the merge operation succeeded")
    message: str = Field(..., description="Detailed message about the merge operation")
    merged_path: Optional[str] = Field(default=None, description="Path to the merged checkpoint if successful")
