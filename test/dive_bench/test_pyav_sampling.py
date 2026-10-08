"""Real CPU video decoding checks; no downloaded videos or model weights."""

import av
import numpy as np
import pytest

from lmms_eval.models.model_utils.grt.load_video import read_video_pyav_seek_uniform


@pytest.mark.parametrize("count,budget", [(31, 8), (3, 8)])
def test_pyav_frame_indices_match_target_contract(tmp_path, count, budget):
    path = tmp_path / "synthetic.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=5)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for index in range(count):
            pixels = np.full((32, 32, 3), index * 7, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    decoded = read_video_pyav_seek_uniform(str(path), num_frm=budget)
    expected = np.linspace(0, count - 1, min(budget, count), dtype=int)
    assert len(decoded) == len(expected)
    # MPEG4/YUV is lossy, but adjacent frame levels remain distinguishable.
    actual = np.rint(decoded.mean(axis=(1, 2, 3)) / 7).astype(int)
    assert actual.tolist() == expected.tolist()
