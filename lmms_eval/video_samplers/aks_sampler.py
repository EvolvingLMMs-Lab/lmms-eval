from .base import BaseVideoSampler
from . import register_video_sampler
from typing import Any, Dict, Optional

@register_video_sampler("aks")
class AKSVideoSampler(BaseVideoSampler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = self.__class__.__name__
        self.extract_feature_model = kwargs.get("extract_feature_model", "blip")
        self.max_num_frames = kwargs.get("max_num_frames", 64)
        self.ratio = kwargs.get("ratio", 1)
        self.t1 = kwargs.get("t1", 0.8)
        self.t2 = kwargs.get("t2", -100)
        self.all_depth = kwargs.get("all_depth", 5)

    def sample(self, ele: Any, **kwargs) -> Optional[Dict[str, Any]]:
        # TODO: Implement AKS sampling
        return ele["video"]