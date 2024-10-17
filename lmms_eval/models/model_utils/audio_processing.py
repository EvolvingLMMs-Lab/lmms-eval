import numpy as np
from librosa import resample


def downsample_audio(audio_array: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
    audio_resample_array = resample(audio_array, orig_sr=original_sr, target_sr=target_sr)
    return audio_resample_array
