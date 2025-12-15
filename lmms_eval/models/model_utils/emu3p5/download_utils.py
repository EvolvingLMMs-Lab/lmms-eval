"""Utility functions for downloading and managing Emu3.5 model weights."""

import os
from pathlib import Path
from typing import Optional

from accelerate import Accelerator
from huggingface_hub import snapshot_download
from loguru import logger as eval_logger


def get_default_cache_dir() -> Path:
    """
    Get default cache directory for model weights.

    Returns:
        Path to cache directory, respecting LMMS_EVAL_MODELS_CACHE env variable
    """
    cache_env = os.getenv("LMMS_EVAL_MODELS_CACHE")
    if cache_env:
        return Path(cache_env).expanduser()
    return Path.home() / ".cache" / "lmms-eval" / "models"


def ensure_local_weights(
    path: str,
    hf_repo_id: str,
    cache_base_dir: Optional[str] = None,
    accelerator: Optional[Accelerator] = None,
) -> str:
    """
    Ensure model weights exist locally, downloading if necessary.

    This function checks if the provided path exists locally. If it does,
    the path is returned. If not, the function downloads the model from
    HuggingFace to a cache directory and returns the cache path.

    Args:
        path: Desired local path to check (can be relative, absolute, or contain ~)
        hf_repo_id: HuggingFace repository ID to download from if path doesn't exist
        cache_base_dir: Base directory for cache. If None, uses default cache location
        accelerator: Accelerator instance for distributed training coordination.
                    If provided, only the main process will download.

    Returns:
        Absolute path to the local weights directory

    Raises:
        OSError: If download fails or cache directory cannot be created

    Example:
        >>> # Path exists locally
        >>> path = ensure_local_weights("/local/emu3.5", "BAAI/Emu3.5")
        >>> # Returns: "/local/emu3.5"
        >>>
        >>> # Path doesn't exist, downloads to cache
        >>> path = ensure_local_weights("nonexistent", "BAAI/Emu3.5")
        >>> # Downloads and returns: "~/.cache/lmms-eval/models/BAAI/Emu3.5"
    """
    # Expand path (handle ~, env vars, relative paths)
    expanded_path = Path(path).expanduser().resolve()

    # Check if path exists locally
    if expanded_path.exists() and expanded_path.is_dir():
        eval_logger.info(
            f"Found local weights at {expanded_path}, using them directly"
        )
        return str(expanded_path)

    # Path doesn't exist, need to download
    eval_logger.info(
        f"Local path {expanded_path} not found, will download from HuggingFace"
    )

    # Determine cache directory
    if cache_base_dir:
        cache_dir = Path(cache_base_dir).expanduser()
    else:
        cache_dir = get_default_cache_dir()

    # Construct cache path based on HF repo ID
    # BAAI/Emu3.5 -> cache_dir/BAAI/Emu3.5
    repo_cache_path = cache_dir / hf_repo_id

    # Check if already in cache
    if repo_cache_path.exists() and repo_cache_path.is_dir():
        eval_logger.info(
            f"Found cached weights at {repo_cache_path}, using them"
        )
        return str(repo_cache_path)

    # Need to download - handle distributed training
    if accelerator and accelerator.num_processes > 1:
        # Distributed training: only main process downloads
        if accelerator.is_main_process:
            eval_logger.info(
                f"Main process downloading {hf_repo_id} to {repo_cache_path}"
            )
            _download_from_hf(hf_repo_id, repo_cache_path)
        else:
            eval_logger.info(
                f"Worker process waiting for main process to download {hf_repo_id}"
            )

        # All processes wait for main process to finish downloading
        accelerator.wait_for_everyone()
        eval_logger.info(f"All processes ready, using weights from {repo_cache_path}")
    else:
        # Single process: just download
        eval_logger.info(f"Downloading {hf_repo_id} to {repo_cache_path}")
        _download_from_hf(hf_repo_id, repo_cache_path)

    return str(repo_cache_path)


def _download_from_hf(hf_repo_id: str, local_dir: Path) -> None:
    """
    Download model weights from HuggingFace Hub.

    Args:
        hf_repo_id: HuggingFace repository ID
        local_dir: Local directory to download to

    Raises:
        OSError: If download fails or directory cannot be created
    """
    try:
        # Create cache directory if it doesn't exist
        local_dir.parent.mkdir(parents=True, exist_ok=True)

        # Download from HuggingFace
        # local_dir_use_symlinks=False to avoid symlink issues
        snapshot_download(
            repo_id=hf_repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
        eval_logger.info(f"Successfully downloaded {hf_repo_id} to {local_dir}")

    except Exception as e:
        eval_logger.error(
            f"Failed to download {hf_repo_id} from HuggingFace: {e}"
        )
        eval_logger.error(
            "Please check your internet connection and verify the repository ID"
        )
        raise
