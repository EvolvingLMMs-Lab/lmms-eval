import os

import numpy as np
from decord import VideoReader, cpu
from loguru import logger as eval_logger
from PIL import Image

from lmms_eval.tasks.vsibench.utils import (
    base_cache_dir,
    cache_name,
)


def vsibench_doc_to_visual_as_images(doc, lmms_eval_specific_kwargs=None):
    """
    Return video frames as a list of PIL Images instead of video path.
    This allows the model to process the video as multi-image input.

    Args:
        doc: Document containing video metadata
        lmms_eval_specific_kwargs: Optional kwargs containing 'num_frames' (default: 32)

    Returns:
        List of PIL.Image objects sampled uniformly from the video
    """
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = doc["dataset"] + "/" + doc["scene_name"] + ".mp4"
    video_path = os.path.join(cache_dir, video_path)

    if not os.path.exists(video_path):
        raise FileExistsError(f"video path:{video_path} does not exist.")

    # Get number of frames from lmms_eval_specific_kwargs or default to 32
    num_frames = 32
    if lmms_eval_specific_kwargs:
        num_frames = lmms_eval_specific_kwargs.get("num_frames", 32)

    # Load video and sample frames uniformly
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)

    # Ensure we don't request more frames than available
    num_frames = min(num_frames, total_frames)

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()

    # Convert to PIL Images
    pil_images = [Image.fromarray(frame) for frame in frames]

    eval_logger.info(f"Loaded {len(pil_images)} frames from video as images (total_frames={total_frames})")

    return pil_images
