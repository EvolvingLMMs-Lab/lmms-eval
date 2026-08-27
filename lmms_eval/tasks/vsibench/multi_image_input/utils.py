import os

import numpy as np
from loguru import logger as eval_logger
from PIL import Image

from lmms_eval.tasks.vsibench.utils import (
    _get_specific_kwarg,
    base_cache_dir,
    cache_name,
    format_vsibench_neo_ov_video_content,
    sample_vsibench_video_frames,
    vsibench_doc_to_text,
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
    from decord import VideoReader, cpu

    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = doc["dataset"] + "/" + doc["scene_name"] + ".mp4"
    video_path = os.path.join(cache_dir, video_path)

    if not os.path.exists(video_path):
        raise FileExistsError(f"video path:{video_path} does not exist.")

    num_frames = int(_get_specific_kwarg(lmms_eval_specific_kwargs, "num_frames", 32))

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


def vsibench_doc_to_messages_as_images(doc, lmms_eval_specific_kwargs=None):
    lmms_eval_specific_kwargs = lmms_eval_specific_kwargs or {}
    prompt = vsibench_doc_to_text(doc, lmms_eval_specific_kwargs)

    if lmms_eval_specific_kwargs.get("prompt_format") == "neo_ov":
        frames = sample_vsibench_video_frames(doc, lmms_eval_specific_kwargs)
        content = format_vsibench_neo_ov_video_content(frames, prompt)
    else:
        frames = vsibench_doc_to_visual_as_images(doc, lmms_eval_specific_kwargs)
        content = [{"type": "image", "url": frame} for frame in frames]
        content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]
