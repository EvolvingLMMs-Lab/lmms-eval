import av
from av.codec.context import CodecContext
import numpy as np


def read_video_pyav(video_path, num_frm=8):
    frames = []
    container = av.open(video_path)

    if "webm" not in video_path and "mkv" not in video_path:
        total_frames = container.streams.video[0].frames
        num_frm = min(total_frames, num_frm)
        indices = np.linspace(0, total_frames - 1, num_frm, dtype=int)
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
    else:
        # https://github.com/PyAV-Org/PyAV/issues/1269
        context = CodecContext.create("libvpx-vp9", "r")
        for packet in container.demux(video=0):
            for frame in context.decode(packet):
                frames.append(frame)
        total_frames = len(frames)
        num_frm = min(total_frames, num_frm)
        indices = np.linspace(0, total_frames - 1, num_frm, dtype=int)
        frames = [frames[i] for i in indices]
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])
