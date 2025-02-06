import base64
from io import BytesIO
from random import sample
from typing import Optional, Tuple, Union

import av
import numpy as np
from av.codec.context import CodecContext
from decord import VideoReader, cpu
from PIL import Image


def load_video_decord(video_path, max_frames_num):
    if type(video_path) == str:
        vr = VideoReader(video_path, ctx=cpu(0))
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames  # (frames, height, width, channels)


# This one is faster
def record_video_length_stream(container, indices):
    frames = []
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
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


def load_video_stream(container, num_frm: int = 8, fps: float = None, force_include_last_frame=False):
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


def load_video_packet(container, num_frm: int = 8, fps: float = None):
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


def read_video_pyav(video_path: str, *, num_frm: int = 8, fps: float = None, format="rgb24", force_include_last_frame=False) -> np.ndarray:
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

    if "webm" not in video_path and "mkv" not in video_path:
        # For mp4, we try loading with stream first
        try:
            frames = load_video_stream(container, num_frm, fps, force_include_last_frame=force_include_last_frame)
        except:
            frames = record_video_length_packet(container)
    else:
        frames = record_video_length_packet(container)

    return np.stack([x.to_ndarray(format=format) for x in frames])


def read_video_pyav_pil(video_path: str, *, num_frm: int = 8, fps: float = None, format="rgb24", max_image_size: Optional[Union[Tuple[int, int], int]] = None, resize_strategy: str = "resize", force_include_last_frame=False):
    frames = read_video_pyav(video_path, num_frm=num_frm, fps=fps, format=format, force_include_last_frame=force_include_last_frame)
    pil_frames = []
    for frame in frames:
        img = Image.fromarray(frame)
        if max_image_size:
            if resize_strategy == "resize":
                if isinstance(max_image_size, int):
                    max_image_size = (max_image_size, max_image_size)
                img = img.resize(max_image_size)
            elif resize_strategy == "thumbnail":
                img.thumbnail(max_image_size)
            else:
                raise ValueError(f"Unknown resize strategy: {resize_strategy}")
        pil_frames.append(img)
    return pil_frames
    # return [Image.fromarray(frame) for frame in frames]


def read_video_pyav_base64(video_path: str, *, num_frm: int = 8, fps: Optional[float] = None, format="rgb24", img_format="PNG", max_image_size: Optional[Union[Tuple[int, int], int]] = None, resize_strategy: str = "resize"):
    frames = read_video_pyav(video_path, num_frm=num_frm, fps=fps, format=format)
    base64_frames = []
    for frame in frames:
        img = Image.fromarray(frame)
        if max_image_size:
            if resize_strategy == "resize":
                if isinstance(max_image_size, int):
                    max_image_size = (max_image_size, max_image_size)
                img = img.resize(max_image_size)
            elif resize_strategy == "thumbnail":
                img.thumbnail(max_image_size)
            else:
                raise ValueError(f"Unknown resize strategy: {resize_strategy}")
        output_buffer = BytesIO()
        img.save(output_buffer, format=img_format)
        byte_data = output_buffer.getvalue()
        base64_str = base64.b64encode(byte_data).decode("utf-8")
        base64_frames.append(base64_str)
    return base64_frames
