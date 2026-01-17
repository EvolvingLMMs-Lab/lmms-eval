import os

import numpy as np
from decord import VideoReader, cpu
from loguru import logger as eval_logger
from PIL import Image

from lmms_eval.tasks.sitebench.utils import (
    UpperLetters,
    base_cache_dir,
    cache_name,
    spatial_aggregate_results,
    spatial_doc_to_text_video,
    spatial_process_results,
)


def spatial_doc_to_visual_video_as_images(doc, lmms_eval_specific_kwargs=None):
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
    video_path = doc["visual"][0]
    video_path = os.path.join(cache_dir, video_path)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video path: {video_path} does not exist.")

    # Get number of frames from lmms_eval_specific_kwargs or default to 32
    num_frames = 32
    if lmms_eval_specific_kwargs:
        num_frames = lmms_eval_specific_kwargs.get("default", {}).get("num_frames", 32)

    # Load video and sample frames uniformly
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)

    # Ensure we don't request more frames than available
    num_frames = min(num_frames, total_frames)

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()

    # Convert to PIL Images
    pil_images = [Image.fromarray(frame) for frame in frames]

    return pil_images


def spatial_doc_to_messages_video_as_images(doc, lmms_eval_specific_kwargs=None):
    """
    Convert a sitebench video document to chat messages format using multi-image input.
    Puts all image frames at the front, then the question text.

    Args:
        doc: Document containing video metadata
        lmms_eval_specific_kwargs: Optional kwargs containing 'num_frames' (default: 32)

    Returns:
        List containing a single message dict with role and content
    """
    question = spatial_doc_to_text_video(doc, lmms_eval_specific_kwargs)
    visuals = spatial_doc_to_visual_video_as_images(doc, lmms_eval_specific_kwargs)

    # Build content as a list with all images first, then text
    content = []
    for visual in visuals:
        content.append({"type": "image", "url": visual})
    content.append({"type": "text", "text": question})

    messages = [{"role": "user", "content": content}]

    return messages
