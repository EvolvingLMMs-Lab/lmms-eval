from __future__ import annotations

from typing import Any, Dict, Optional


import torch

from .base import BaseVideoSampler
from . import register_video_sampler

from qwen_vl_utils.vision_process import smart_nframes

import random
from PIL import Image

import sys
from external.longclip.model import longclip

@register_video_sampler("qframe")
class QFrameVideoSampler(BaseVideoSampler):
    """Sampler that samples frames uniformly from a video."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = self.__class__.__name__
        self.max_num_frames = kwargs.get("max_num_frames", None)
        self.model_path = kwargs.get("qframe_model_path", "./Long-CLIP/checkpoints/longclip-L.pt")
        self.clip_model, self.clip_processor = longclip.load(self.model_path, device=self.device)
        self.tau = kwargs.get("tau", 0.8)
        self.high_frames = kwargs.get("high_frames", 6)
        self.mid_frames = kwargs.get("mid_frames", 6)
        self.low_frames = kwargs.get("low_frames", 8)

    def text_image_matching(self, question, images, tau=1.0):

        # print(f"{text}\n{'-'*100}\n{question}")
        with torch.no_grad(), torch.cuda.amp.autocast():
            text = longclip.tokenize([question]).to(self.device)
            images = torch.stack([self.clip_processor(Image.fromarray(image)) for image in images]).to(self.device)
            
            image_features = self.clip_model.encode_image(images)
            text_features = self.clip_model.encode_text(text)
            logits_per_text = text_features @ image_features.T  # this is the image-text similarity score

        probs = (logits_per_text / tau).softmax(dim=1)[0]
        
        probs = torch.log(probs) - torch.log(-torch.log(torch.rand(len(images), device=probs.device) + 1e-10) + 1e-10)  # gumble

        indices = np.argsort(-probs.cpu().detach().numpy())

        return indices

    def load_video(self, ele, max_num_frames, fps=1, force_sample=False):
        if max_num_frames == 0:
            return np.zeros((1, 336, 336, 3))
        vr = ele["video_reader"]
        total_frame_num = ele["total_frames"]
        video_fps = ele["video_fps"]
        video_time = total_frame_num / video_fps
        fps = round(video_fps / fps)
        frame_idx = [i for i in range(0, total_frame_num, fps)]
        frame_time = [i / fps for i in frame_idx]
        if len(frame_idx) > max_num_frames or force_sample:
            sample_fps = max_num_frames
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i / video_fps for i in frame_idx]
        frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        return spare_frames, frame_idx, frame_time, video_time

    def sample(
        self,
        ele: Any,
        **kwargs
    ) -> Tuple[List[int], int]:
        visual, frame_idx, frame_time, video_time = self.load_video(ele, self.max_num_frames)

        indices = self.text_image_matching(ele["question"], visual, tau=self.tau)
        visual_tmp = [None] * len(visual)
        visual = [Image.fromarray(v).convert("RGB") for v in visual]
        width, height = visual[0].size
        for idx in indices[:self.high_frames]:
            visual_tmp[idx] = visual[idx].resize((width // 2, height // 2), Image.Resampling.LANCZOS)
        for idx in indices[self.high_frames: self.high_frames+self.mid_frames]:
            visual_tmp[idx] =visual[idx].resize((width // 4, height // 4), Image.Resampling.LANCZOS)
        for idx in indices[self.high_frames+self.mid_frames: self.high_frames+self.mid_frames+self.low_frames]:
            visual_tmp[idx] =visual[idx].resize((width // 8, height // 8), Image.Resampling.LANCZOS)
        
        visual = [v for v in visual_tmp if v is not None ]
    
        
        return idx

