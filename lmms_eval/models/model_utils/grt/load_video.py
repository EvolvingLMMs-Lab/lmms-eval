import base64
from io import BytesIO
from typing import Optional, Tuple, Union

import av
import numpy as np
from PIL import Image

from lmms_eval.imports import optional_import


def _missing_decord(*args, **kwargs):
    raise ImportError("The selected Decord backend requires decord; use pyav_seek for the verified LPM profiles.")


VideoReader, _has_decord = optional_import("decord", "VideoReader", fallback=_missing_decord)
cpu, _ = optional_import("decord", "cpu", fallback=_missing_decord)


def load_video_decord(video_path, max_frames_num):
    if type(video_path) is str:
        vr = VideoReader(video_path, ctx=cpu(0))
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames  # (frames, height, width, channels)


def _uniform_frame_indices(total_frames: int, max_frames_num: int, fps: Optional[float] = None, avg_fps: Optional[float] = None):
    if total_frames <= 0:
        raise ValueError("Cannot sample frames from an empty video")

    sample_count = max_frames_num
    if fps is not None and avg_fps is not None and avg_fps > 0:
        duration_s = total_frames / avg_fps
        sample_count = min(max_frames_num, max(1, int(duration_s * fps)))

    sample_count = min(max(1, int(sample_count)), total_frames)
    return np.linspace(0, total_frames - 1, sample_count, dtype=int).tolist()


def read_video_decord_uniform(video_path: str, *, num_frm: int = 8, fps: Optional[float] = None) -> np.ndarray:
    path = video_path if isinstance(video_path, str) else video_path[0]
    vr = VideoReader(path, ctx=cpu(0))
    total_frames = len(vr)
    avg_fps = float(vr.get_avg_fps()) if hasattr(vr, "get_avg_fps") else None
    frame_idx = _uniform_frame_indices(total_frames, num_frm, fps=fps, avg_fps=avg_fps)
    return vr.get_batch(frame_idx).asnumpy()


def _stream_total_frames(stream, avg_fps: Optional[float]) -> int:
    if stream.frames:
        return int(stream.frames)
    if stream.duration and avg_fps and stream.time_base:
        return max(1, int(float(stream.duration * stream.time_base) * avg_fps))
    raise ValueError("Cannot determine video frame count")


def _frame_index_from_pts(frame, stream, avg_fps: float) -> Optional[int]:
    if frame.pts is None or stream.time_base is None:
        return None
    start_time = stream.start_time or 0
    seconds = float((frame.pts - start_time) * stream.time_base)
    return int(round(seconds * avg_fps))


def read_video_pyav_seek_uniform(
    video_path: str,
    *,
    num_frm: int = 8,
    fps: Optional[float] = None,
    format="rgb24",
    force_include_last_frame=False,
) -> np.ndarray:
    path = video_path if isinstance(video_path, str) else video_path[0]
    container = av.open(path)
    try:
        stream = container.streams.video[0]
        avg_fps = float(stream.average_rate) if stream.average_rate else None
        if not avg_fps or avg_fps <= 0:
            raise ValueError("Cannot determine video FPS")

        total_frames = _stream_total_frames(stream, avg_fps)
        frame_indices = _uniform_frame_indices(total_frames, num_frm, fps=fps, avg_fps=avg_fps)
        if force_include_last_frame and frame_indices and frame_indices[-1] != total_frames - 1:
            frame_indices[-1] = total_frames - 1
        start_time = stream.start_time or 0
        frames = []
        for target_index in frame_indices:
            target_seconds = target_index / avg_fps
            if stream.time_base:
                seek_offset = start_time + int(target_seconds / float(stream.time_base))
                container.seek(seek_offset, any_frame=False, backward=True, stream=stream)
            else:
                container.seek(int(target_seconds * av.time_base), any_frame=False, backward=True)

            selected = None
            fallback = None
            for frame in container.decode(stream):
                fallback = frame
                frame_index = _frame_index_from_pts(frame, stream, avg_fps)
                if frame_index is None or frame_index >= target_index:
                    selected = frame
                    break
            if selected is None:
                if fallback is None:
                    raise ValueError(f"Could not decode frame near index {target_index} from {path}")
                selected = fallback
            frames.append(selected)

        return np.stack([frame.to_ndarray(format=format) for frame in frames])
    finally:
        container.close()


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

    try:
        return read_video_pyav_seek_uniform(
            video_path,
            num_frm=num_frm,
            fps=fps,
            format=format,
            force_include_last_frame=force_include_last_frame,
        )
    except Exception:
        pass

    container = av.open(video_path)

    if "webm" not in video_path and "mkv" not in video_path:
        # For mp4, we try loading with stream first
        try:
            frames = load_video_stream(container, num_frm, fps, force_include_last_frame=force_include_last_frame)
        except Exception:
            frames = record_video_length_packet(container)
    else:
        frames = record_video_length_packet(container)

    return np.stack([x.to_ndarray(format=format) for x in frames])


def record_video_frames_ds(container):
    """
    Decode all frames from the video container and return
    them as a list. Works for any codec by demuxing + decoding.
    """
    frames = []
    for packet in container.demux(video=0):
        for frame in packet.decode():
            frames.append(frame)
    return frames


def load_video_stream_ds(container, num_frm: int = 8, sample_fps: float = None, force_include_last_frame: bool = False):
    """
    Sample frames from the video using streaming decode.

    Args:
        container: an av.Container opened via av.open().
        num_frm:   maximum number of frames if sample_fps is None.
        sample_fps: target sampling rate in frames per second. If set,
                    overrides num_frm to sample approximately duration*sample_fps frames.
        force_include_last_frame: ensure the very last frame is included in the sample.

    Returns:
        List[av.VideoFrame]: the sampled frames.
    """
    # get total frame count and stream FPS
    total_frames = container.streams.video[0].frames
    stream_fps = float(container.streams.video[0].average_rate)
    # compute video duration in seconds
    duration = total_frames / stream_fps

    # decide how many frames to sample
    if sample_fps is not None:
        # sample approximately duration * sample_fps frames
        desired_count = int(duration * sample_fps)
    else:
        desired_count = num_frm
    # clamp to at least 1 and at most total_frames
    sample_count = min(max(desired_count, 1), total_frames)

    # uniformly pick indices across the video
    indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)
    if force_include_last_frame:
        last_idx = total_frames - 1
        if last_idx not in indices:
            # replace the final index to guarantee inclusion
            indices = np.concatenate([indices[:-1], [last_idx]])

    # decode all frames then select
    all_frames = record_video_frames_ds(container)
    return [all_frames[i] for i in indices]


def load_video_packet_ds(container, num_frm: int = 8, sample_fps: float = None):
    """
    Decode all frames at once, then sample based on num_frm or sample_fps.

    Args:
        container: an av.Container opened via av.open().
        num_frm:   maximum number of frames if sample_fps is None.
        sample_fps: target sampling rate in frames per second.

    Returns:
        List[av.VideoFrame]: the sampled frames.
    """
    # decode everything first
    all_frames = record_video_frames_ds(container)
    total_frames = len(all_frames)
    stream_fps = float(container.streams.video[0].average_rate)
    duration = total_frames / stream_fps

    if sample_fps is not None:
        desired_count = int(duration * sample_fps)
    else:
        desired_count = num_frm
    sample_count = min(max(desired_count, 1), total_frames)

    indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)
    # always include last frame if it's missing
    if total_frames - 1 not in indices:
        indices = np.append(indices, total_frames - 1)

    return [all_frames[i] for i in indices]


def read_video_pyav_ds(video_path: str, num_frm: int = 8, sample_fps: float = None, format: str = "rgb24", force_include_last_frame: bool = False) -> np.ndarray:
    """
    Read video and return sampled frames as a NumPy array.

    Args:
        video_path: path to the video file.
        num_frm:    max frames if sample_fps is None.
        sample_fps: target FPS for sampling.
        format:     pixel format for to_ndarray().
        force_include_last_frame: ensure the last frame is in the result.

    Returns:
        np.ndarray of shape (n_selected, H, W, C).
    """
    container = av.open(video_path)
    try:
        # primary: stream‐based sampling
        frames = load_video_stream_ds(container, num_frm, sample_fps, force_include_last_frame)
    except Exception:
        # fallback: packet‐based sampling
        frames = load_video_packet_ds(container, num_frm, sample_fps)
    # convert VideoFrame → ndarray
    arrs = [f.to_ndarray(format=format) for f in frames]
    return np.stack(arrs)


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


def read_video_decord_base64(video_path: str, *, num_frm: int = 8, fps: Optional[float] = None, img_format="PNG", max_image_size: Optional[Union[Tuple[int, int], int]] = None, resize_strategy: str = "resize"):
    frames = read_video_decord_uniform(video_path, num_frm=num_frm, fps=fps)
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


def read_video_pyav_seek_base64(video_path: str, *, num_frm: int = 8, fps: Optional[float] = None, format="rgb24", img_format="PNG", max_image_size: Optional[Union[Tuple[int, int], int]] = None, resize_strategy: str = "resize"):
    frames = read_video_pyav_seek_uniform(video_path, num_frm=num_frm, fps=fps, format=format)
    return frames_to_base64(frames, img_format=img_format, max_image_size=max_image_size, resize_strategy=resize_strategy)


def frames_to_base64(frames: np.ndarray, *, img_format="PNG", max_image_size: Optional[Union[Tuple[int, int], int]] = None, resize_strategy: str = "resize"):
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
