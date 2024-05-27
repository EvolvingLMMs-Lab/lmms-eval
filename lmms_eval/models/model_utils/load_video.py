import av
import numpy as np

def read_video_pyav(video_path, num_frm=8):
    frames = []
    container = av.open(video_path)

    # sample uniformly 8 frames from the video
    total_frames = container.streams.video[0].frames
    num_frm = min(total_frames, num_frm)
    indices = np.arange(0, total_frames, total_frames / num_frm).astype(int)
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])