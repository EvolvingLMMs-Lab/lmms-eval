import importlib
import os
from typing import Optional, Union

import numpy as np

SUPPORTED_VIDEO_DECODE_BACKENDS = ("pyav", "torchcodec", "dali", "decord")


def _resolve_video_path(video_path: Union[str, tuple, list]) -> str:
    if isinstance(video_path, str):
        return video_path
    if isinstance(video_path, (tuple, list)) and len(video_path) > 0 and isinstance(video_path[0], str):
        return video_path[0]
    raise TypeError(f"Unsupported video_path type: {type(video_path).__name__}")


def _normalize_decode_backend(backend: Optional[str]) -> str:
    selected = (backend or os.getenv("LMMS_VIDEO_DECODE_BACKEND", "pyav")).strip().lower()
    if selected not in SUPPORTED_VIDEO_DECODE_BACKENDS:
        expected = ", ".join(SUPPORTED_VIDEO_DECODE_BACKENDS)
        raise ValueError(f"Unsupported video decode backend: {selected}. Expected one of: {expected}")
    return selected


def _import_pyav():
    try:
        return importlib.import_module("av")
    except ModuleNotFoundError as exc:
        raise ImportError("PyAV backend requires `av`. Install it via `uv add av`.") from exc


def _import_decord():
    try:
        return importlib.import_module("decord")
    except ModuleNotFoundError as exc:
        raise ImportError("Decord backend requires the legacy video extra. Install via `uv sync --extra video-legacy`.") from exc


def _probe_video_metadata(video_path: str) -> tuple[int, Optional[float]]:
    av = _import_pyav()
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
        raise ValueError("Cannot sample a video with no decoded frames")
    if num_frm <= 0:
        raise ValueError("num_frm must be greater than zero")
    if fps is not None and fps <= 0:
        raise ValueError("fps must be greater than zero when provided")

    sampled = min(total_frames, num_frm)
    if fps is not None and frame_rate and frame_rate > 0:
        video_length = total_frames / frame_rate
        sampled = min(sampled, int(video_length * fps))
    return max(1, sampled)


def _compute_uniform_indices(total_frames: int, sampled_frm: int, force_include_last_frame: bool = False) -> np.ndarray:
    if total_frames <= 0:
        raise ValueError("total_frames must be > 0")
    if sampled_frm <= 0 or sampled_frm > total_frames:
        raise ValueError("sampled_frm must be between 1 and total_frames")

    if sampled_frm == 1:
        index = total_frames - 1 if force_include_last_frame else 0
        return np.array([index], dtype=int)

    # With sampled_frm <= total_frames, these indices are unique and include
    # both endpoints. Keeping one index policy across random-access backends
    # makes decoder comparisons and evaluation reproduction easier.
    return np.linspace(0, total_frames - 1, sampled_frm, dtype=int)


def _open_decord_reader(video_path: Union[str, tuple, list]):
    decord = _import_decord()
    resolved_path = _resolve_video_path(video_path)
    num_threads = int(os.getenv("LMMS_VIDEO_DECORD_THREADS", "2"))
    return decord.VideoReader(resolved_path, ctx=decord.cpu(0), num_threads=num_threads)


def load_video_decord(video_path, max_frames_num):
    """Legacy Decord helper that always returns ``max_frames_num`` frames.

    This intentionally retains duplicate indices for short videos because a
    few model adapters rely on that historical behavior. New code should use
    :func:`read_video` with ``backend="decord"`` for maximum-count semantics.
    """

    if max_frames_num <= 0:
        raise ValueError("max_frames_num must be greater than zero")

    vr = _open_decord_reader(video_path)
    try:
        total_frame_num = len(vr)
        if total_frame_num <= 0:
            raise ValueError("Cannot decode frames from an empty video")
        frame_idx = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int).tolist()
        return np.ascontiguousarray(vr.get_batch(frame_idx).asnumpy())
    finally:
        del vr  # Release VideoReader before interpreter teardown.


def read_video_decord(
    video_path: Union[str, tuple, list],
    *,
    num_frm: int = 8,
    fps: Optional[float] = None,
    format="rgb24",
    force_include_last_frame=False,
) -> np.ndarray:
    if format != "rgb24":
        raise ValueError("Decord backend currently supports format='rgb24' only")

    vr = _open_decord_reader(video_path)
    try:
        total_frames = len(vr)
        get_avg_fps = getattr(vr, "get_avg_fps", None)
        frame_rate = float(get_avg_fps()) if get_avg_fps is not None else None
        sampled_frm = _compute_sample_count(total_frames, num_frm, fps, frame_rate)
        indices = _compute_uniform_indices(total_frames, sampled_frm, force_include_last_frame=force_include_last_frame)
        frames = vr.get_batch(indices.tolist()).asnumpy()
        if frames.dtype != np.uint8:
            frames = frames.astype(np.uint8)
        return np.ascontiguousarray(frames)
    finally:
        del vr  # Release VideoReader before interpreter teardown.


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
    stream = container.streams.video[0]
    total_frames = int(stream.frames or 0)
    if total_frames <= 0:
        raise ValueError("Video stream does not report a reliable frame count")

    frame_rate = float(stream.average_rate) if stream.average_rate is not None else None
    sampled_frm = _compute_sample_count(total_frames, num_frm, fps, frame_rate)
    indices = _compute_uniform_indices(total_frames, sampled_frm, force_include_last_frame=force_include_last_frame)
    frames = record_video_length_stream(container, indices)
    if len(frames) != len(indices):
        raise RuntimeError(f"Video metadata reported {total_frames} frames, but only {len(frames)} sampled frames could be decoded")
    return frames


def load_video_packet(container, num_frm: int = 8, fps: Optional[float] = None, force_include_last_frame=False):
    frames = record_video_length_packet(container)
    total_frames = len(frames)
    stream = container.streams.video[0]
    frame_rate = float(stream.average_rate) if stream.average_rate is not None else None
    sampled_frm = _compute_sample_count(total_frames, num_frm, fps, frame_rate)
    indices = _compute_uniform_indices(total_frames, sampled_frm, force_include_last_frame=force_include_last_frame)
    return [frames[i] for i in indices]


def _frames_to_ndarray(frames, format: str) -> np.ndarray:
    if not frames:
        raise ValueError("Cannot decode frames from an empty video")

    converted = [frame.to_ndarray(format=format) for frame in frames]
    try:
        output = np.stack(converted)
    except ValueError as exc:
        shapes = [frame.shape for frame in converted]
        raise ValueError(f"Decoded video frames have inconsistent shapes: {shapes}") from exc
    return np.ascontiguousarray(output)


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
    ``pyav`` (default), ``torchcodec``, ``dali``, and legacy ``decord``.

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
    if selected_backend == "decord":
        return read_video_decord(
            resolved_path,
            num_frm=num_frm,
            fps=fps,
            format=format,
            force_include_last_frame=force_include_last_frame,
        )

    return read_video_pyav(
        resolved_path,
        num_frm=num_frm,
        fps=fps,
        format=format,
        force_include_last_frame=force_include_last_frame,
    )


def read_video_pyav(
    video_path: Union[str, tuple, list],
    *,
    num_frm: int = 8,
    fps: Optional[float] = None,
    format="rgb24",
    force_include_last_frame=False,
) -> np.ndarray:
    """Decode with PyAV, falling back to a full packet scan when needed."""

    resolved_path = _resolve_video_path(video_path)
    av = _import_pyav()
    container = av.open(resolved_path)

    try:
        container.streams.video[0].thread_type = "AUTO"
        packet_scan_required = resolved_path.lower().split("?", maxsplit=1)[0].endswith((".webm", ".mkv"))
        if not packet_scan_required:
            try:
                frames = load_video_stream(
                    container,
                    num_frm,
                    fps,
                    force_include_last_frame=force_include_last_frame,
                )
            except Exception:
                # Reopen instead of seeking: a failed decode may leave the
                # codec context in a state that is not reset by seek(0).
                container.close()
                container = av.open(resolved_path)
                container.streams.video[0].thread_type = "AUTO"
                frames = load_video_packet(
                    container,
                    num_frm,
                    fps,
                    force_include_last_frame=force_include_last_frame,
                )
        else:
            frames = load_video_packet(
                container,
                num_frm,
                fps,
                force_include_last_frame=force_include_last_frame,
            )
        return _frames_to_ndarray(frames, format)
    finally:
        container.close()  # Ensure container is closed to prevent resource leak
