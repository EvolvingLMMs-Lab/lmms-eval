from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from librosa import resample
from loguru import logger as eval_logger


def get_datasets_version() -> Tuple[int, int, int]:
    """Get the installed datasets library version as a tuple."""
    try:
        import datasets

        version_str = datasets.__version__
        parts = version_str.split(".")
        major = int(parts[0]) if len(parts) > 0 else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2].split("+")[0].split("-")[0]) if len(parts) > 2 else 0
        return (major, minor, patch)
    except Exception:
        return (0, 0, 0)


def load_audio_safe(
    doc: Dict[str, Any],
    audio_key: str = "audio",
    target_sr: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """
    Load audio data with version-aware handling for datasets >= 4.0.

    The datasets library version 4.0+ uses torchcodec for audio loading,
    which can cause compatibility issues with certain audio formats.
    This function provides a unified interface that handles both old
    and new datasets formats.

    Args:
        doc: Document dictionary containing audio data
        audio_key: Key to access audio data in the document
        target_sr: Target sample rate for resampling (optional)

    Returns:
        Tuple of (audio_array, sample_rate)

    Raises:
        ValueError: If audio data cannot be loaded or is in unexpected format
    """
    audio_data = doc.get(audio_key)
    if audio_data is None:
        raise ValueError(f"Audio key '{audio_key}' not found in document")

    datasets_version = get_datasets_version()
    audio_array: np.ndarray
    sample_rate: int

    try:
        if datasets_version[0] >= 4:
            if hasattr(audio_data, "numpy"):
                audio_array = audio_data.numpy()
                sample_rate = getattr(audio_data, "sampling_rate", 16000)
            elif isinstance(audio_data, dict):
                audio_array = np.array(audio_data["array"])
                sample_rate = audio_data.get("sampling_rate", 16000)
            elif isinstance(audio_data, np.ndarray):
                audio_array = audio_data
                sample_rate = 16000
            else:
                audio_array = np.array(audio_data)
                sample_rate = 16000
                eval_logger.warning(f"Unexpected audio format with datasets {datasets_version}, " f"attempting conversion. Type: {type(audio_data)}")
        else:
            if isinstance(audio_data, dict):
                audio_array = np.array(audio_data["array"])
                sample_rate = audio_data.get("sampling_rate", 16000)
            else:
                audio_array = np.array(audio_data)
                sample_rate = 16000

        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=-1)

        if target_sr is not None and target_sr != sample_rate:
            audio_array = resample(audio_array, orig_sr=sample_rate, target_sr=target_sr)
            sample_rate = target_sr

        return audio_array, sample_rate

    except Exception as e:
        raise ValueError(f"Failed to load audio from document. datasets version: {datasets_version}, " f"audio_data type: {type(audio_data)}, error: {e}") from e


def downsample_audio(audio_array: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
    audio_resample_array = resample(audio_array, orig_sr=original_sr, target_sr=target_sr)
    return audio_resample_array


def split_audio(audio_arrays: np.ndarray, chunk_lim: int) -> List:
    """
    Splits the audio into chunks of a given length.
    Args:
        audio_arrays: The audio array to split.
        chunk_lim: The length of each chunk.
    Returns:
        A list of audio chunks.
    """
    audio_splits = []
    # Split the loaded audio to 30s chunks and extend the messages content
    for i in range(
        0,
        len(audio_arrays),
        chunk_lim,
    ):
        audio_splits.append(audio_arrays[i : i + chunk_lim])
    return audio_splits
