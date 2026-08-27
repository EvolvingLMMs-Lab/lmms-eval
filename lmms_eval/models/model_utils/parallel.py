"""Shared launcher-parallelism helpers for model backends."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ParallelPlan:
    """Launcher-level topology read from torch.distributed env vars."""

    global_rank: int
    local_rank: int
    world_size: int
    tp_size: int = 1

    @classmethod
    def from_env(cls, tp_size: int = 1) -> "ParallelPlan":
        return cls(
            global_rank=int(os.environ.get("RANK", 0)),
            local_rank=int(os.environ.get("LOCAL_RANK", 0)),
            world_size=int(os.environ.get("WORLD_SIZE", 1)),
            tp_size=int(tp_size),
        )

    @property
    def dp_size(self) -> int:
        return max(1, self.world_size // max(1, self.tp_size))

    def device_str(self) -> str:
        import torch

        if not torch.cuda.is_available():
            return "cpu"
        return f"cuda:{self.local_rank}"


def resolve_local_rank_device(device: str, plan: ParallelPlan | None = None) -> str:
    """Pin distributed CUDA launches to LOCAL_RANK while leaving single-process args alone."""

    plan = plan or ParallelPlan.from_env()
    device = str(device)
    if plan.world_size <= 1 or not device.startswith("cuda"):
        return device

    import torch

    device_count = torch.cuda.device_count()
    if device_count <= 0:
        return device
    return f"cuda:{plan.local_rank % device_count}"
