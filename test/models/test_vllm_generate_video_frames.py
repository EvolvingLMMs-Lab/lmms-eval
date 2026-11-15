"""Regression tests for vllm_generate video frame layout.

These checks stay on CPU. They do not run vLLM or a video model.
"""

import numpy as np
import pytest
import torch

from lmms_eval.models.chat.vllm_generate import _as_uint8_thwc


def _paint_channels(shape, channels_first: bool) -> np.ndarray:
    """Fill each channel with a distinct value so a transpose cannot hide a swap."""
    frames = np.zeros(shape, dtype=np.float32)
    channel_axis = 1 if channels_first else -1
    for channel, value in enumerate((10, 80, 200, 40)):
        if channel >= shape[channel_axis]:
            break
        index = [slice(None)] * frames.ndim
        index[channel_axis] = channel
        frames[tuple(index)] = value
    return frames


def test_tchw_float_becomes_uint8_thwc_and_keeps_channels():
    frames = _paint_channels((2, 3, 4, 5), channels_first=True)

    converted = _as_uint8_thwc(frames)

    assert converted.shape == (2, 4, 5, 3)
    assert converted.dtype == np.uint8
    assert converted.flags.c_contiguous
    np.testing.assert_array_equal(converted[..., 0], 10)
    np.testing.assert_array_equal(converted[..., 1], 80)
    np.testing.assert_array_equal(converted[..., 2], 200)


def test_torch_tchw_tensor_uses_the_same_layout():
    frames = torch.tensor(_paint_channels((1, 3, 2, 6), channels_first=True))

    converted = _as_uint8_thwc(frames)

    assert converted.shape == (1, 2, 6, 3)
    np.testing.assert_array_equal(converted[0, 0, 0], np.array([10, 80, 200], dtype=np.uint8))


@pytest.mark.parametrize("channels", [1, 4])
def test_other_channel_counts_transpose_when_the_last_axis_is_spatial(channels):
    shape = (2, channels, 3, 5)
    frames = _paint_channels(shape, channels_first=True)

    converted = _as_uint8_thwc(frames)

    assert converted.shape == (2, 3, 5, channels)
    for channel in range(channels):
        expected = (10, 80, 200, 40)[channel]
        np.testing.assert_array_equal(converted[..., channel], expected)


def test_thwc_uint8_passes_through_without_changing_channels():
    frames = _paint_channels((2, 4, 5, 3), channels_first=False).astype(np.uint8)

    converted = _as_uint8_thwc(frames)

    assert converted.shape == frames.shape
    np.testing.assert_array_equal(converted, frames)


def test_clipping_keeps_in_range_channel_values_and_saturates_the_rest():
    # Already channels-last, so the heuristic must not transpose before clipping.
    frames = np.zeros((1, 2, 2, 3), dtype=np.float32)
    frames[..., 0] = -3.2
    frames[..., 1] = 10.6
    frames[..., 2] = 300.4

    converted = _as_uint8_thwc(frames)

    assert converted.shape == (1, 2, 2, 3)
    np.testing.assert_array_equal(converted[..., 0], 0)
    np.testing.assert_array_equal(converted[..., 1], 11)
    np.testing.assert_array_equal(converted[..., 2], 255)


def test_non_video_rank_is_rejected():
    with pytest.raises(ValueError, match="4-D"):
        _as_uint8_thwc(np.zeros((3, 8, 8), dtype=np.uint8))
