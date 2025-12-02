# Infeasible for long vidoes because of the memory usage
from __future__ import annotations

from typing import Any, List, Tuple

import imageio
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
import cv2
from multiprocessing.dummy import Pool as ThreadPool
import random

from .base import BaseVideoSampler
from . import register_video_sampler

from qwen_vl_utils.vision_process import smart_nframes

@register_video_sampler("mgsampler")
class MGVideoSampler(BaseVideoSampler):
    """Sample frames from the video.
    Required keys are "filename", "total_frames", "start_index" , added or
    modified keys are "frame_inds", "frame_interval" and "num_clips".
    Args:
        clip_len (int): Frames of each sampled output clip.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Default: 1.
        num_clips (int): Number of clips to be sampled. Default: 1.
        temporal_jitter (bool): Whether to apply temporal jittering.
            Default: False.
        twice_sample (bool): Whether to use twice sample when testing.
            If set to True, it will sample frames with and without fixed shift,
            which is commonly used for testing in TSM model. Default: False.
        out_of_bound_opt (str): The way to deal with out of bounds frame
            indexes. Available options are 'loop', 'repeat_last'.
            Default: 'loop'.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        start_index (None): This argument is deprecated and moved to dataset
            class (``BaseDataset``, ``VideoDatset``, ``RawframeDataset``, etc),
            see this: https://github.com/open-mmlab/mmaction2/pull/89.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = self.__class__.__name__
        self.num_frames = kwargs.get("num_frames", None)
        self.test_mode = kwargs.get("test_mode", True)

    def multiplication(self,video_path):

        img = list()
        img_diff = []
        try:

            vid = imageio.get_reader(video_path, 'ffmpeg')
            for num, im in enumerate[Any](vid):
                img.append(im)
            for i in range(len(img) - 1):
                tmp1 = cv2.cvtColor(img[i], cv2.COLOR_RGB2GRAY)
                tmp2 = cv2.cvtColor(img[i + 1], cv2.COLOR_RGB2GRAY)
                (score, diff) = compare_ssim(tmp1, tmp2, full=True)
                score = 1 - score
                img_diff.append(score)
        except(OSError):
            video_name = (video_path.split('/')[-1]).split('.')[0]
            raise ValueError(f"error! {video_name}")
        return img_diff

    def sample(
        self,
        ele: Any,
        **kwargs
    ) -> Tuple[List[int], int]:
        video_path = ele["video"]

        def find_nearest(array, value):
            array = np.asarray(array)
            try:
                idx = (np.abs(array - value)).argmin()
                return int(idx + 1)
            except(ValueError):
                raise ValueError(f"error! {video_path}")

        diff_score = self.multiplication(video_path)
        diff_score = np.power(diff_score, 0.5)
        sum_num = np.sum(diff_score)
        diff_score = diff_score / sum_num

        count = 0
        pic_diff = list()
        for i in range(len(diff_score)):
            count = count + diff_score[i]
            pic_diff.append(count)

        choose_index = list()

        if self.test_mode:
            choose_index.append(find_nearest(pic_diff, 1 / 32))
            choose_index.append(find_nearest(pic_diff, 1 / 32 + 1 / 16))
            choose_index.append(find_nearest(pic_diff, 1 / 32 + 2 / 16))
            choose_index.append(find_nearest(pic_diff, 1 / 32 + 3 / 16))
            choose_index.append(find_nearest(pic_diff, 1 / 32 + 4 / 16))
            choose_index.append(find_nearest(pic_diff, 1 / 32 + 5 / 16))
            choose_index.append(find_nearest(pic_diff, 1 / 32 + 6 / 16))
            choose_index.append(find_nearest(pic_diff, 1 / 32 + 7 / 16))
            choose_index.append(find_nearest(pic_diff, 1 / 32 + 8 / 16))
            choose_index.append(find_nearest(pic_diff, 1 / 32 + 9 / 16))
            choose_index.append(find_nearest(pic_diff, 1 / 32 + 10 / 16))
            choose_index.append(find_nearest(pic_diff, 1 / 32 + 11 / 16))
            choose_index.append(find_nearest(pic_diff, 1 / 32 + 12 / 16))
            choose_index.append(find_nearest(pic_diff, 1 / 32 + 13 / 16))
            choose_index.append(find_nearest(pic_diff, 1 / 32 + 14 / 16))
            choose_index.append(find_nearest(pic_diff, 1 / 32 + 15 / 16))

        else:
            choose_index.append(find_nearest(pic_diff, random.uniform(0, 1 / 16)))
            choose_index.append(find_nearest(pic_diff, random.uniform(1 / 16, 2 / 16)))
            choose_index.append(find_nearest(pic_diff, random.uniform(2 / 16, 3 / 16)))
            choose_index.append(find_nearest(pic_diff, random.uniform(3 / 16, 4 / 16)))
            choose_index.append(find_nearest(pic_diff, random.uniform(4 / 16, 5 / 16)))
            choose_index.append(find_nearest(pic_diff, random.uniform(5 / 16, 6 / 16)))
            choose_index.append(find_nearest(pic_diff, random.uniform(6 / 16, 7 / 16)))
            choose_index.append(find_nearest(pic_diff, random.uniform(7 / 16, 8 / 16)))
            choose_index.append(find_nearest(pic_diff, random.uniform(8 / 16, 9 / 16)))
            choose_index.append(find_nearest(pic_diff, random.uniform(9 / 16, 10 / 16)))
            choose_index.append(find_nearest(pic_diff, random.uniform(10 / 16, 11 / 16)))
            choose_index.append(find_nearest(pic_diff, random.uniform(11 / 16, 12 / 16)))
            choose_index.append(find_nearest(pic_diff, random.uniform(12 / 16, 13 / 16)))
            choose_index.append(find_nearest(pic_diff, random.uniform(13 / 16, 14 / 16)))
            choose_index.append(find_nearest(pic_diff, random.uniform(14 / 16, 15 / 16)))
            choose_index.append(find_nearest(pic_diff, random.uniform(15 / 16, 16 / 16)))

        return choose_index

    

    
        

