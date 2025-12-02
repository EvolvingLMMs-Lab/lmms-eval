from __future__ import annotations

from typing import Any, Dict, Optional

import sys
sys.path.append("./external/D-KTS")
import torchvision.models as models
from feature_extractor import dhash, hanming
import torch
from torch.autograd import Variable
import numpy as np
from kts.auto_alter import cpd_auto2
from kts.nonlin_alter import kernel

from .base import BaseVideoSampler
from . import register_video_sampler

from qwen_vl_utils.vision_process import smart_nframes

@register_video_sampler("dkts")
class DKTSVideoSampler(BaseVideoSampler):
    """Sampler that samples frames uniformly from a video."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = self.__class__.__name__
        self.threshold = kwargs.get("threshold", 4.0)
        self.v = kwargs.get("v", 1.0) # vmax in penalty
        self.use_cpu = kwargs.get("use_cpu", False)
        self.extract_frequency = kwargs.get("extract_frequency", 1)
        googlenet = models.googlenet(pretrained=True)
        self.googlenet = torch.nn.Sequential(*list(googlenet.children())[:-2])
        self.googlenet.eval()
        if not self.use_cpu:
            self.googlenet = self.googlenet.cuda()
        else:
            print('Using CPU......')

    def get_frames(self, vr, frame_num, backend): # (T, C, H, W)
        if backend == 'torchcodec':
            full_raw_image_tensors = vr.get_frames_at(indices=frame_num).data
        elif backend == 'decord':
            full_raw_image_tensors = torch.from_numpy(vr.get_batch(frame_num).asnumpy().permute(0, 3, 1, 2))
        elif backend == 'torchvision':
            full_raw_image_tensors = vr[frame_num]
        else:
            raise ValueError(f"backend {backend} not supported")
        return full_raw_image_tensors

    def get_features(self, ele: Any): # from one_video_hash
        vr = ele['video_reader']
        fps = ele['video_fps']
        frame_count = ele["total_frames"]
        if self.extract_frequency == "fps":
            self.extract_frequency = int(fps)

        frames=[]
        video_features = []
        count = 0
        skip_count = 0
        arr = []

        with torch.no_grad():
            base = None
            hash1 = None
            for count in range(0, frame_count, self.extract_frequency):
                if count % self.extract_frequency == 0:
                    fr = self.get_frames(vr, [count], ele['video_reader_backend'])[0].permute(1,2,0).numpy()
                    hash2=dhash(fr)
                    if hash1 is not None:
                        dist = hanming(hash1,hash2)
                    if base is None or dist > self.threshold:
                        base = fr
                        hash1 = hash2
                        frames.append(np.rollaxis(fr, 2))
                        arr.append(skip_count)
                        skip_count = 0
                    else:
                        skip_count += 1
                        frames.append(np.rollaxis(base, 2))
                    if (len(frames) == self.batch_size) or (count >= frame_count and len(frames) > 0):
                        batch = np.array(frames)
                        if self.use_cpu:
                            variable = Variable(torch.from_numpy(batch).float())
                            feature = self.googlenet(variable).detach().numpy()
                        else:
                            variable = Variable(torch.from_numpy(batch).float()).cuda()
                            feature = self.googlenet(variable).cpu().detach().numpy()
                        video_features.extend(feature)
                        frames.clear()
        video_features = np.squeeze(np.array(video_features))
        duration = frame_count/fps
        picks = np.arange(0, video_features.shape[0]) * self.extract_frequency
        return {
            "n_frames": int(frame_count),
            "features": video_features,
            "picks": picks,
            "duration": duration,
            "skip_arr": arr,
        }

    def kts_run(self, ele: Any):
        features = self.get_features(ele)
        X = features['features']
        n_frames = features['n_frames']
        n = X.shape[0]
        n1 = min(n, 338) # 95%
        m = round(n_frames / 106 * 2)

        K1 = kernel(X, X.T, n1)
        cps1, scores1 = cpd_auto2(K1, m, self.v,self.extract_frequency)
        cps1 *= self.extract_frequency
        cps1 = np.hstack((0, cps1, n_frames))
        begin_frames = cps1[:-1]
        end_frames = cps1[1:]
        cps1 = np.vstack((begin_frames, end_frames - 1)).T
        return cps1

    def sample(
        self,
        ele: Any,
        **kwargs
    ) -> Tuple[List[int], int]:
        cps = self.kts_run(ele)
        frame_indices = []
        for i in range(cps.shape[0]):
            start_frame = cps[i, 0]
            end_frame = cps[i, 1]
            frame_idx = int((start_frame+end_frame)/2)
            frame_indices.append(frame_idx)
        return frame_indices

