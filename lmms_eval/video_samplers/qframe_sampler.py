from __future__ import annotations

from typing import Any, Dict, Optional


import torch

from .base import BaseVideoSampler
from . import register_video_sampler

from qwen_vl_utils.vision_process import smart_nframes, smart_resize

import random
from PIL import Image

import sys
import cv2
from io import BytesIO
import base64
from external.longclip.model import longclip
import numpy as np
from typing import List, Tuple
import decord

@register_video_sampler("qframe")
class QFrameVideoSampler(BaseVideoSampler):
    """Sampler that samples frames uniformly from a video."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = self.__class__.__name__
        self.num_frames = int(kwargs.get("num_frames", 32))
        self.model_path = kwargs.get("qframe_model_path", "external/longclip/checkpoints/longclip-L.pt")
        self.clip_model, self.clip_processor = longclip.load(self.model_path, device=self.device)
        self.tau = float(kwargs.get("tau", 0.8))
        self.high_frames = int(kwargs.get("high_frames", 6))
        self.mid_frames = int(kwargs.get("mid_frames", 6))
        self.low_frames = int(kwargs.get("low_frames", 8))
        self.return_frames = True
        self.baseline = kwargs.get("baseline", False)
        self.will_process_messages = True

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

    def load_video(self, video_path, num_frames, fps=1, force_sample=False):
        if num_frames == 0:
            return np.zeros((1, 336, 336, 3))
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)
        total_frame_num = len(vr)
        video_time = total_frame_num / vr.get_avg_fps()
        fps = round(vr.get_avg_fps() / fps)
        frame_idx = [i for i in range(0, len(vr), fps)]
        frame_time = [i / fps for i in frame_idx]
        if len(frame_idx) > num_frames or force_sample:
            sample_fps = num_frames
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i / vr.get_avg_fps() for i in frame_idx]
        frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
        spare_frames = vr.get_batch(frame_idx).numpy()

        video_metadata = dict(
            fps=vr.get_avg_fps(),
            frames_indices=frame_idx,
            total_num_frames=len(vr),
            video_backend="decord",
        )

        return spare_frames, frame_idx, frame_time, video_time, video_metadata

    def process_messages(self, chat_message, eval_logger):
        messages = chat_message[0]["content"]
        new_messages = []
        for i, message in enumerate(messages):
            if message["type"] in ["video", "image"]:
                visual = message["url"]
                question = message["question"]
                if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):  # Video file
                    # modify the video processing to multi-image processing
                    """
                    vr = decord.VideoReader(visual)
                    first_frame = vr[0].asnumpy()
                    height, width = first_frame.shape[:2]
                    # max_pixels = height * width
                    message.append({"role": "user", "content": [{"type": "video", "video": visual, "max_pixels": self.max_pixels}, {"type": "text", "text": context}]})
                    """
                    visual, frame_idx, frame_time, video_time, video_metadata = self.load_video(visual, self.num_frames)

                    try:
                        indices = self.text_image_matching(question, visual, tau=self.tau)

                        visual = [Image.fromarray(v).convert("RGB") for v in visual]
                        if not self.baseline:
                            visual_tmp = [None] * len(visual)
                            width, height = visual[0].size
                            for idx in indices[:self.high_frames]:
                                visual_tmp[idx] = visual[idx].resize((width // 2, height // 2), Image.Resampling.LANCZOS)
                            for idx in indices[self.high_frames: self.high_frames+self.mid_frames]:
                                visual_tmp[idx] =visual[idx].resize((width // 4, height // 4), Image.Resampling.LANCZOS)
                            for idx in indices[self.high_frames+self.mid_frames: self.high_frames+self.mid_frames+self.low_frames]:
                                visual_tmp[idx] =visual[idx].resize((width // 8, height // 8), Image.Resampling.LANCZOS)
                            visual = [v for v in visual_tmp if v is not None ]
                        else:
                            visual_tmp = [None] * len(visual)
                            for idx in indices:
                                visual_tmp[idx] = visual[idx]
                            visual = [v for v in visual_tmp if v is not None ]
                    except Exception as e:
                        eval_logger.info(f"{e}")
                        if len(visual) >= self.sample_frames:
                            visual = visual[sorted(random.sample(range(len(visual)), self.sample_frames))]
                        height, width, _ = visual[0].shape
                        visual = [Image.fromarray(v).convert("RGB").resize((width // 2, height // 2), Image.Resampling.LANCZOS) for v in visual]
                    
                    image_content = []
                    for base64_image in visual:
                        # base64_image = Image.fromarray(v).convert("RGB")
                        buffer = BytesIO()
                        base64_image.save(buffer, format="JPEG")
                        base64_bytes = base64.b64encode(buffer.getvalue())
                        base64_string = base64_bytes.decode("utf-8")
                        new_messages.append({"type": "image", "url": f"data:image/jpeg;base64,{base64_string}"})

                elif isinstance(visual, Image.Image):  # Single image
                    base64_image = visual.convert("RGB")
                    buffer = BytesIO()
                    base64_image.save(buffer, format="JPEG")
                    base64_bytes = base64.b64encode(buffer.getvalue())
                    base64_string = base64_bytes.decode("utf-8")
                    new_messages.append({"type": "image", "url": f"data:image/jpeg;base64,{base64_string}"})
                elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):  # Multiple images
                    for v in visual:
                        base64_image = v.convert("RGB")
                        buffer = BytesIO()
                        base64_image.save(buffer, format="JPEG")
                        base64_bytes = base64.b64encode(buffer.getvalue())
                        base64_string = base64_bytes.decode("utf-8")
                        new_messages.append({"type": "image", "url": f"data:image/jpeg;base64,{base64_string}"})
                else:
                    raise ValueError(f"Invalid visual type: {type(visual)}")
            else:
                new_messages.append(message)
        chat_message[0]["content"] = new_messages
        return chat_message, video_metadata

    def sample(
        self,
        ele: Any,
        **kwargs
    ) -> Tuple[List[int], int]:
        if self.num_frames is not None:
            ele["nframes"] = self.num_frames
        nframes = smart_nframes(ele, total_frames=ele["total_frames"], video_fps=ele["video_fps"])
        idx = torch.linspace(ele["start_frame"], ele["end_frame"], nframes).round().long().tolist()
        return idx

    # def load_video(self, ele, max_num_frames, fps=1, force_sample=False):
    #     if max_num_frames == 0:
    #         return np.zeros((1, 336, 336, 3))
    #     vr = ele["video_reader"]
    #     total_frame_num = ele["total_frames"]
    #     video_fps = ele["video_fps"]
    #     video_time = total_frame_num / video_fps
    #     fps = round(video_fps / fps)
    #     frame_idx = [i for i in range(0, total_frame_num, fps)]
    #     frame_time = [i / fps for i in frame_idx]
    #     if len(frame_idx) > max_num_frames or force_sample:
    #         sample_fps = max_num_frames
    #         uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
    #         frame_idx = uniform_sampled_frames.tolist()
    #         frame_time = [i / video_fps for i in frame_idx]
    #     frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    #     spare_frames = vr.get_batch(frame_idx).asnumpy()
    #     return spare_frames, frame_idx, frame_time, video_time


    # def sample(
    #     self,
    #     ele: Any,
    #     **kwargs
    # ) -> Tuple[List[int], int]:
    #     visual, frame_idx, frame_time, video_time = self.load_video(ele, self.max_num_frames)

    #     indices = self.text_image_matching(ele["question"], visual, tau=self.tau)

    #     if ele["video_reader_backend"] == "decord":
    #         T, H, W, C = visual.shape
    #     else:
    #         T, C, H, W = visual.shape
    #     visual_tmp = [None] * len(visual)
    #     for idx in indices[:self.high_frames]:
    #         visual_tmp[idx] = cv2.resize(visual[idx], (W//2, H//2), interpolation=cv2.INTER_LANCZOS4)

    #     for idx in indices[self.high_frames:self.high_frames+self.mid_frames]:
    #         visual_tmp[idx] = cv2.resize(visual[idx], (W//4, H//4), interpolation=cv2.INTER_LANCZOS4)

    #     for idx in indices[self.high_frames+self.mid_frames:self.high_frames+self.mid_frames+self.low_frames]:
    #         visual_tmp[idx] = cv2.resize(visual[idx], (W//8, H//8), interpolation=cv2.INTER_LANCZOS4)

    #     # TODO DEBUG THIS
    #     patch_factor = int(image_patch_size * SPATIAL_MERGE_SIZE)
    #     for v in visual_tmp:
    #         if v is None:
    #             continue
    #         if "resized_height" in ele and "resized_width" in ele:
    #             resized_height, resized_width = smart_resize(
    #                 ele["resized_height"],
    #                 ele["resized_width"],
    #                 factor=patch_factor,
    #             )
    #         else:
    #             width, height = W, H
    #             min_pixels = ele.get("min_pixels", IMAGE_MIN_TOKEN_NUM * patch_factor ** 2)
    #             max_pixels = ele.get("max_pixels", IMAGE_MAX_TOKEN_NUM * patch_factor ** 2)
    #             resized_height, resized_width = smart_resize(
    #                 height,
    #                 width,
    #                 factor=patch_factor,
    #                 min_pixels=min_pixels,
    #                 max_pixels=max_pixels,
    #             )
    #         image = cv2.resize((resized_width, resized_height))
    #         visual.append(image)

    #     nframes = smart_nframes(ele, total_frames=ele["total_frames"], video_fps=ele["video_fps"])
    #     idx = torch.linspace(ele["start_frame"], ele["end_frame"], nframes).round().long().tolist()
    #     return idx, visual

