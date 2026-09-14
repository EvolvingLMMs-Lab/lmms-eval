"""Strict one-GPU worker for the pinned, same-device GRT profiles."""

from __future__ import annotations

import json
import os
import random
import socket
import sys
from argparse import Namespace
from importlib.metadata import version
from pathlib import Path

_PROCESS_SIZE_ENV = ("WORLD_SIZE", "LOCAL_WORLD_SIZE", "PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "MV2_COMM_WORLD_SIZE", "SLURM_NTASKS", "SLURM_NPROCS")
_PROCESS_RANK_ENV = ("RANK", "LOCAL_RANK", "PMI_RANK", "OMPI_COMM_WORLD_RANK", "MV2_COMM_WORLD_RANK", "SLURM_PROCID")


def _require_single_process(torch_module) -> None:
    """Reject distributed launch metadata and an already initialized process group."""
    for name in (*_PROCESS_SIZE_ENV, *_PROCESS_RANK_ENV):
        raw = os.environ.get(name)
        if raw is None:
            continue
        try:
            value = int(raw)
        except ValueError as exc:
            raise RuntimeError(f"Invalid {name}; GRT profiles require a single process.") from exc
        allowed = (1,) if name in _PROCESS_SIZE_ENV else ((-1, 0) if name == "LOCAL_RANK" else (0,))
        if value not in allowed:
            raise RuntimeError(f"{name}={value}; GRT profiles require a single process, not a distributed launcher.")
    if torch_module.distributed.is_available() and torch_module.distributed.is_initialized():
        if torch_module.distributed.get_world_size() != 1:
            raise RuntimeError("An initialized distributed process group violates the single-process GRT contract.")


def main() -> None:
    """Run native lmms-eval after setting the historical determinism policy."""
    if os.getenv("PYTHONHASHSEED") != "0" or os.getenv("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Set PYTHONHASHSEED=0 and CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python.")
    if any(arg == "--config" or arg.startswith("--config=") for arg in sys.argv[1:]):
        raise RuntimeError("The strict GRT worker accepts explicit profile flags, not --config files that can override its environment.")
    if "--output_path" not in sys.argv:
        raise RuntimeError("A new --output_path is required to preserve run provenance.")
    output = Path(sys.argv[sys.argv.index("--output_path") + 1])
    if output.exists():
        raise FileExistsError(f"Refusing to reuse an existing run directory: {output}")
    import numpy as np
    import torch

    _require_single_process(torch)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one CUDA GPU; run base/all/candidate serially on that device.")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print(
        "[DIVE_RUNTIME] "
        + json.dumps(
            {
                "hostname": socket.gethostname(),
                "gpu": torch.cuda.get_device_name(0),
                "gpu_uuid": str(getattr(torch.cuda.get_device_properties(0), "uuid", "")),
                "cuda": torch.version.cuda,
                "python": sys.version,
                "bootstrap_seed": 0,
                "packages": {name: version(name) for name in ("torch", "torchvision", "transformers", "qwen-vl-utils", "av", "accelerate")},
            }
        ),
        flush=True,
    )
    from lmms_eval.__main__ import cli_evaluate

    # Upstream intentionally catches evaluation exceptions at INFO verbosity.
    # Its DEBUG branch re-raises them; force that documented native CLI boundary
    # even if the caller supplied a quieter verbosity. A failed inference must
    # yield a nonzero process exit, never an apparently successful GRT run.
    cli_evaluate(args=Namespace(verbosity="DEBUG"))


if __name__ == "__main__":
    main()
