"""Utilities for model memory profiling and reporting."""

from typing import Optional

import torch
from accelerate import Accelerator
from loguru import logger as eval_logger


def get_model_size_gb(model: torch.nn.Module) -> float:
    """
    Calculate model size in GB based on parameters and their dtype.

    Args:
        model: PyTorch model (can be wrapped or unwrapped)

    Returns:
        Model size in GB
    """
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_bytes / (1024**3)


def print_memory_stats(
    main_model: torch.nn.Module,
    image_tokenizer: torch.nn.Module,
    accelerator: Optional[Accelerator] = None,
    device_idx: Optional[int] = None,
    rank: int = 0,
) -> None:
    """
    Print model sizes and GPU memory usage statistics.

    Args:
        main_model: Main model (can be accelerator-wrapped)
        image_tokenizer: Image tokenizer model (can be accelerator-wrapped)
        accelerator: Accelerator instance for unwrapping models (optional)
        device_idx: CUDA device index for memory stats (optional)
        rank: Process rank for multi-GPU setups
    """
    # Unwrap models if wrapped by accelerator
    unwrapped_main_model = (
        accelerator.unwrap_model(main_model) if accelerator is not None else main_model
    )
    unwrapped_image_tokenizer = (
        accelerator.unwrap_model(image_tokenizer)
        if accelerator is not None
        else image_tokenizer
    )

    # Calculate model sizes
    main_model_size_gb = get_model_size_gb(unwrapped_main_model)
    image_tokenizer_size_gb = get_model_size_gb(unwrapped_image_tokenizer)
    total_model_size_gb = main_model_size_gb + image_tokenizer_size_gb

    eval_logger.info(f"[Rank {rank}] Main model size: {main_model_size_gb:.2f} GB")
    eval_logger.info(
        f"[Rank {rank}] Image tokenizer size: {image_tokenizer_size_gb:.2f} GB"
    )
    eval_logger.info(f"[Rank {rank}] Total model size: {total_model_size_gb:.2f} GB")

    # Get GPU memory usage if available
    if torch.cuda.is_available():
        # Use provided device_idx or default to 0
        device = device_idx if device_idx is not None else 0
        allocated_gb = torch.cuda.memory_allocated(device) / (1024**3)
        reserved_gb = torch.cuda.memory_reserved(device) / (1024**3)
        total_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)

        allocated_pct = (allocated_gb / total_gb) * 100
        reserved_pct = (reserved_gb / total_gb) * 100

        eval_logger.info(
            f"[Rank {rank}] GPU Memory Allocated: "
            f"{allocated_gb:.2f} GB / {total_gb:.2f} GB ({allocated_pct:.1f}%)"
        )
        eval_logger.info(
            f"[Rank {rank}] GPU Memory Reserved: "
            f"{reserved_gb:.2f} GB / {total_gb:.2f} GB ({reserved_pct:.1f}%)"
        )
    else:
        eval_logger.warning(
            f"[Rank {rank}] CUDA not available, cannot report GPU memory usage"
        )
