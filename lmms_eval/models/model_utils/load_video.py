import importlib
import os
from typing import Optional, Union

import av
import numpy as np


def _resolve_video_path(video_path: Union[str, tuple, list]) -> str:
    if isinstance(video_path, str):
        return video_path
    if isinstance(video_path, (tuple, list)) and len(video_path) > 0 and isinstance(video_path[0], str):
        return video_path[0]
    raise TypeError(f"Unsupported video_path type: {type(video_path).__name__}")


def _normalize_decode_backend(backend: Optional[str]) -> str:
    selected = (backend or os.getenv("LMMS_VIDEO_DECODE_BACKEND", "pyav")).strip().lower()
    if selected not in {"pyav", "torchcodec", "dali"}:
        raise ValueError(f"Unsupported video decode backend: {selected}. Expected one of: pyav, torchcodec, dali")
    return selected


def _probe_video_metadata(video_path: str) -> tuple[int, Optional[float]]:
    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        total_frames = int(stream.frames or 0)
        frame_rate = float(stream.average_rate) if stream.average_rate is not None else None
        return total_frames, frame_rate
    finally:
        container.close()


def _compute_sample_count(total_frames: int, num_frm: int, fps: Optional[float], frame_rate: Optional[float]) -> int:
    if total_frames <= 0:
        return max(1, num_frm)
    sampled = min(total_frames, num_frm)
    if fps is not None and frame_rate and frame_rate > 0:
        video_length = total_frames / frame_rate
        sampled = min(sampled, int(video_length * fps))
    return max(1, sampled)


def _compute_uniform_indices(total_frames: int, sampled_frm: int, force_include_last_frame: bool = False) -> np.ndarray:
    if total_frames <= 0:
        raise ValueError("total_frames must be > 0")
    indices = np.linspace(0, total_frames - 1, sampled_frm, dtype=int)
    if force_include_last_frame:
        last_frame = total_frames - 1
        if last_frame not in indices:
            if sampled_frm <= 1:
                indices = np.array([last_frame], dtype=int)
            else:
                base = np.linspace(0, max(0, total_frames - 2), sampled_frm - 1, dtype=int)
                indices = np.append(base, last_frame)
    return np.unique(indices)


def load_video_decord(video_path, max_frames_num):
    try:
        decord = importlib.import_module("decord")
    except ModuleNotFoundError as exc:
        raise ImportError("load_video_decord requires `decord`. Install decord to use this backend.") from exc

    VideoReader = decord.VideoReader
    cpu = decord.cpu
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


def read_video_torchcodec(
    video_path: Union[str, tuple, list],
    *,
    num_frm: int = 8,
    fps: Optional[float] = None,
    format="rgb24",
    force_include_last_frame=False,
) -> np.ndarray:
    resolved_path = _resolve_video_path(video_path)
    if format != "rgb24":
        raise ValueError("TorchCodec backend currently supports format='rgb24' only")

    try:
        decoders = importlib.import_module("torchcodec.decoders")
    except ModuleNotFoundError as exc:
        raise ImportError("TorchCodec backend requires `torchcodec`. Install via `uv add torchcodec`.") from exc

    VideoDecoder = decoders.VideoDecoder
    threads_raw = os.getenv("LMMS_VIDEO_TORCHCODEC_THREADS", "0")
    try:
        threads = int(threads_raw)
    except ValueError:
        threads = 0

    decoder_kwargs = {
        "device": os.getenv("LMMS_VIDEO_TORCHCODEC_DEVICE", "cpu"),
        "dimension_order": "NHWC",
    }
    if threads > 0:
        decoder_kwargs["num_ffmpeg_threads"] = threads

    decoder = VideoDecoder(resolved_path, **decoder_kwargs)

    metadata = getattr(decoder, "metadata", None)
    total_frames = int(getattr(metadata, "num_frames", 0) or 0)
    frame_rate = getattr(metadata, "average_fps", None)
    frame_rate = float(frame_rate) if frame_rate is not None else None

    if total_frames <= 0:
        total_frames, fallback_fps = _probe_video_metadata(resolved_path)
        if frame_rate is None:
            frame_rate = fallback_fps

    sampled_frm = _compute_sample_count(total_frames, num_frm, fps, frame_rate)
    indices = _compute_uniform_indices(total_frames, sampled_frm, force_include_last_frame=force_include_last_frame)
    frames_batch = decoder.get_frames_at(indices.tolist())
    data = frames_batch.data if hasattr(frames_batch, "data") else frames_batch

    if hasattr(data, "cpu"):
        data = data.cpu()
    frames = data.numpy() if hasattr(data, "numpy") else np.asarray(data)

    if frames.ndim != 4:
        raise ValueError(f"Unexpected TorchCodec frame tensor shape: {getattr(frames, 'shape', None)}")

    if frames.shape[-1] != 3 and frames.shape[1] == 3:
        frames = np.transpose(frames, (0, 2, 3, 1))
    if frames.dtype != np.uint8:
        frames = frames.astype(np.uint8)
    return np.ascontiguousarray(frames)


def read_video_dali(
    video_path: Union[str, tuple, list],
    *,
    num_frm: int = 8,
    fps: Optional[float] = None,
    format="rgb24",
    force_include_last_frame=False,
) -> np.ndarray:
    resolved_path = _resolve_video_path(video_path)
    if format != "rgb24":
        raise ValueError("DALI backend currently supports format='rgb24' only")

    try:
        dali_fn = importlib.import_module("nvidia.dali.fn")
        dali_types = importlib.import_module("nvidia.dali.types")
        pipeline_def = importlib.import_module("nvidia.dali").pipeline_def
    except ModuleNotFoundError as exc:
        raise ImportError("DALI backend requires `nvidia-dali`. Install a matching build for your CUDA/runtime.") from exc

    total_frames, frame_rate = _probe_video_metadata(resolved_path)
    sampled_frm = _compute_sample_count(total_frames, num_frm, fps, frame_rate)

    stride = max(1, int((total_frames - 1) / max(1, sampled_frm - 1))) if total_frames > 1 else 1
    _ = force_include_last_frame

    device = os.getenv("LMMS_VIDEO_DALI_DEVICE", "gpu").strip().lower()
    if device != "gpu":
        raise ValueError("LMMS_VIDEO_DALI_DEVICE must be 'gpu' for fn.readers.video")
    num_threads = int(os.getenv("LMMS_VIDEO_DALI_THREADS", "2"))
    device_id = int(os.getenv("LMMS_VIDEO_DALI_DEVICE_ID", "0"))

    @pipeline_def
    def _video_pipe(input_path: str):
        video = dali_fn.readers.video(
            device=device,
            filenames=[input_path],
            sequence_length=sampled_frm,
            stride=stride,
            random_shuffle=False,
            image_type=dali_types.RGB,
            dtype=dali_types.UINT8,
            initial_fill=1,
            prefetch_queue_depth=1,
        )
        return video

    pipe = _video_pipe(batch_size=1, num_threads=max(1, num_threads), device_id=device_id, input_path=resolved_path)
    pipe.build()
    out = pipe.run()[0]
    frames = out.as_cpu().as_array()[0]

    if frames.ndim != 4:
        raise ValueError(f"Unexpected DALI frame tensor shape: {getattr(frames, 'shape', None)}")
    if frames.dtype != np.uint8:
        frames = frames.astype(np.uint8)
    return np.ascontiguousarray(frames)


def read_video(
    video_path: Union[str, tuple, list],
    *,
    num_frm: int = 8,
    fps: Optional[float] = None,
    format="rgb24",
    force_include_last_frame=False,
    backend: Optional[str] = None,
) -> np.ndarray:
    """
    Read and uniformly sample video frames.

    Dispatches to the decode backend selected by *backend* (or the
    ``LMMS_VIDEO_DECODE_BACKEND`` env-var).  Supported backends:
    ``pyav`` (default), ``torchcodec``, ``dali``.

    Args:
        video_path: Path to the video file.
        num_frm: Maximum number of frames to extract.
        fps: Target sample rate.  When *None*, *num_frm* frames are
            sampled uniformly over the full duration.
        format: Pixel format passed to the decoder (default ``rgb24``).
        force_include_last_frame: Guarantee the last frame is included.
        backend: Explicit backend override.
    Returns:
        np.ndarray: ``(N, H, W, 3)`` uint8 array of sampled frames.
    """

    resolved_path = _resolve_video_path(video_path)
    selected_backend = _normalize_decode_backend(backend)
    if selected_backend == "torchcodec":
        return read_video_torchcodec(
            resolved_path,
            num_frm=num_frm,
            fps=fps,
            format=format,
            force_include_last_frame=force_include_last_frame,
        )
    if selected_backend == "dali":
        return read_video_dali(
            resolved_path,
            num_frm=num_frm,
            fps=fps,
            format=format,
            force_include_last_frame=force_include_last_frame,
        )

    container = av.open(resolved_path)
    container.streams.video[0].thread_type = "AUTO"

    try:
        if "webm" not in resolved_path and "mkv" not in resolved_path:
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


# Backward-compat alias; will be removed in a future release.
read_video_pyav = read_video
