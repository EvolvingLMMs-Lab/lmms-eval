from __future__ import annotations

from typing import Any, Dict, Optional


import torch

from .base import BaseVideoSampler
from . import register_video_sampler

from qwen_vl_utils.vision_process import smart_nframes

@register_video_sampler("fps")
class FPSVideoSampler(BaseVideoSampler):
    """Sampler that samples frames by fps from a video."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = self.__class__.__name__
        self.fps = kwargs.get("fps", 1)

    def sample(
        self,
        ele: Any,
        **kwargs
    ) -> Tuple[List[int], int]:
        del ele["nframes"]
        ele["fps"] = self.fps
        nframes = smart_nframes(ele, total_frames=ele["total_frames"], video_fps=self.fps)
        idx = torch.linspace(ele["start_frame"], ele["end_frame"], nframes).round().long().tolist()
        return idx

