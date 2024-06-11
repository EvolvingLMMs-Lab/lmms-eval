import av
from av.codec.context import CodecContext
import numpy as np

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
    # context = CodecContext.create("libvpx-vp9", "r")
    for packet in container.demux(video=0):
        for frame in packet.decode(): 
            frames.append(frame)
    return frames

def read_video_pyav(video_path, num_frm=8):
    container = av.open(video_path)

    frames = record_video_length_packet(container)
    total_frames = len(frames)
    num_frm = min(total_frames, num_frm)
    indices = np.linspace(0, total_frames - 1, num_frm, dtype=int)
    frames = [frames[i] for i in indices]
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])
