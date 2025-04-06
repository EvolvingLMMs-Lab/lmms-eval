from typing import List

import numpy as np
from librosa import resample


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
