import os
from typing import Optional, Tuple, Union

import av
import numpy as np
from decord import VideoReader, cpu
from PIL import Image

from lmms_eval.models.model_utils.media_encoder import encode_image_to_base64


def load_video_decord(video_path, max_frames_num):
    num_threads = int(os.getenv("LMMS_VIDEO_DECORD_THREADS", "2"))
    if isinstance(video_path, str):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=num_threads)
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0), num_threads=num_threads)
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    del vr  # Release VideoReader to prevent memory leak
    return spare_frames  # (frames, height, width, channels)


# This one is faster
def record_video_length_stream(container, indices):
    frames = []
    start_index = indices[0]
    end_index = indices[-1]
    index_set = set(indices.tolist() if hasattr(indices, "tolist") else indices)
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in index_set:
            frames.append(frame)
    return frames


# This one works for all types of video
def record_video_length_packet(container):
    frames = []
    # https://github.com/PyAV-Org/PyAV/issues/1269
    # https://www.cnblogs.com/beyond-tester/p/17641872.html
    # context = CodecContext.create("libvpx-vp9", "r")
    for packet in container.demux(video=0):
        for frame in packet.decode():
            frames.append(frame)
    return frames


def load_video_stream(container, num_frm: int = 8, fps: Optional[float] = None, force_include_last_frame=False):
    # container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    frame_rate = container.streams.video[0].average_rate
    if fps is not None:
        video_length = total_frames / frame_rate
        num_frm = min(num_frm, int(video_length * fps))
    sampled_frm = min(total_frames, num_frm)
    indices = np.linspace(0, total_frames - 1, sampled_frm, dtype=int)
    if force_include_last_frame:
        last_frame = total_frames - 1
        if last_frame not in indices:
            indices = np.linspace(0, total_frames - 2, sampled_frm - 1, dtype=int)
            indices = np.append(indices, last_frame)

    return record_video_length_stream(container, indices)


def load_video_packet(container, num_frm: int = 8, fps: Optional[float] = None):
    frames = record_video_length_packet(container)
    total_frames = len(frames)
    frame_rate = container.streams.video[0].average_rate
    if fps is not None:
        video_length = total_frames / frame_rate
        num_frm = min(num_frm, int(video_length * fps))
    sampled_frm = min(total_frames, num_frm)
    indices = np.linspace(0, total_frames - 1, sampled_frm, dtype=int)

    # Append the last frame index if not already included
    if total_frames - 1 not in indices:
        indices = np.append(indices, total_frames - 1)

    return [frames[i] for i in indices]


def read_video_pyav(
    video_path: str,
    *,
    num_frm: int = 8,
    fps: Optional[float] = None,
    format="rgb24",
    force_include_last_frame=False,
) -> np.ndarray:
    """
    Read video using the PyAV library.

    Args:
        video_path (str): The path to the video file.
        num_frm (int, optional): The maximum number of frames to extract. Defaults to 8.
        fps (float, optional): The frames per second for extraction. If `None`, the maximum number of frames will be extracted. Defaults to None.
        format (str, optional): The format of the extracted frames. Defaults to "rgb24".

    Returns:
        np.ndarray: A numpy array containing the extracted frames in RGB format.
    """

    container = av.open(video_path)
    container.streams.video[0].thread_type = "AUTO"

    try:
        if "webm" not in video_path and "mkv" not in video_path:
            # For mp4, we try loading with stream first
            try:
                frames = load_video_stream(
                    container,
                    num_frm,
                    fps,
                    force_include_last_frame=force_include_last_frame,
                )
            except Exception:
                container.seek(0)
                frames = record_video_length_packet(container)
        else:
            frames = record_video_length_packet(container)
        first = frames[0].to_ndarray(format=format)
        output = np.empty((len(frames),) + first.shape, dtype=first.dtype)
        output[0] = first
        for i, frame in enumerate(frames[1:], start=1):
            output[i] = frame.to_ndarray(format=format)
        return output
    finally:
        container.close()  # Ensure container is closed to prevent resource leak


def read_video_pyav_pil(
    video_path: str,
    *,
    num_frm: int = 8,
    fps: Optional[float] = None,
    format="rgb24",
    max_image_size: Optional[Union[Tuple[int, int], int]] = None,
    resize_strategy: str = "resize",
    force_include_last_frame=False,
):
    frames = read_video_pyav(video_path, num_frm=num_frm, fps=fps, format=format, force_include_last_frame=force_include_last_frame)
    pil_frames = []

    def _resize_image(img: Image.Image) -> Image.Image:
        if not max_image_size:
            return img
        if resize_strategy == "resize":
            target = (max_image_size, max_image_size) if isinstance(max_image_size, int) else max_image_size
            scale = min(target[0] / img.width, target[1] / img.height)
            new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
            if new_size != img.size:
                return img.resize(new_size, Image.Resampling.BILINEAR)
            return img
        if resize_strategy == "thumbnail":
            target = (max_image_size, max_image_size) if isinstance(max_image_size, int) else max_image_size
            img.thumbnail(target)
            return img
        raise ValueError(f"Unknown resize strategy: {resize_strategy}")

    for frame in frames:
        pil_frames.append(_resize_image(Image.fromarray(frame)))
    return pil_frames
    # return [Image.fromarray(frame) for frame in frames]


def read_video_pyav_base64(
    video_path: str,
    *,
    num_frm: int = 8,
    fps: Optional[float] = None,
    format="rgb24",
    img_format="JPEG",
    max_image_size: Optional[Union[Tuple[int, int], int]] = None,
    resize_strategy: str = "resize",
):
    container = av.open(video_path)
    container.streams.video[0].thread_type = "AUTO"
    try:
        if "webm" not in video_path and "mkv" not in video_path:
            try:
                frames = load_video_stream(container, num_frm, fps)
            except Exception:
                container.seek(0)
                frames = record_video_length_packet(container)
        else:
            frames = record_video_length_packet(container)
    finally:
        container.close()

    base64_frames = []

    def _resize_image(img: Image.Image) -> Image.Image:
        if not max_image_size:
            return img
        if resize_strategy == "resize":
            target = (max_image_size, max_image_size) if isinstance(max_image_size, int) else max_image_size
            scale = min(target[0] / img.width, target[1] / img.height)
            new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
            if new_size != img.size:
                return img.resize(new_size, Image.Resampling.BILINEAR)
            return img
        if resize_strategy == "thumbnail":
            target = (max_image_size, max_image_size) if isinstance(max_image_size, int) else max_image_size
            img.thumbnail(target)
            return img
        raise ValueError(f"Unknown resize strategy: {resize_strategy}")

    for frame in frames:
        if isinstance(frame, av.VideoFrame):
            img = frame.to_image()
        else:
            img = Image.fromarray(frame if isinstance(frame, np.ndarray) else frame.to_ndarray(format=format))
        img = _resize_image(img)
        base64_frames.append(
            encode_image_to_base64(
                img,
                image_format=img_format,
                convert_rgb=img_format.upper() in {"JPEG", "JPG", "WEBP"},
                quality=85 if img_format.upper() in {"JPEG", "JPG", "WEBP"} else None,
                copy_if_pil=False,
                use_path_cache=False,
            )
        )
    return base64_frames
