from io import BytesIO
from copy import deepcopy
import numpy as np
import os
import base64
from typing import List, Tuple
from tqdm import tqdm
import requests as url_requests
import time
import logging

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval import utils

from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState

try:
    from decord import VideoReader, AudioReader, cpu
except ImportError:
    pass

from PIL import Image

eval_logger = logging.getLogger("lmms-eval")

NUM_SECONDS_TO_SLEEP = 5
API_TYPE = os.getenv("API_TYPE", "openai")
if API_TYPE == "openai":  # FIXME: Please modify this to support other type of API
    API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
else:
    eval_logger.error("API_TYPE not supported")


@register_model("gemini_api")
class GeminiAPI(lmms):
    def __init__(
        self,
        model_version: str = "gpt-4-turbo",  # FIXME: Please modify this to support real gemini model version
        modality: str = "video",
        max_frames_for_video: int = -1,
        frame_rate: int = -1,
        timeout: int = 120,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_version = model_version
        self.modality = modality
        self.max_frames_for_video = max_frames_for_video
        self.frame_rate = frame_rate
        assert self.modality in ["image", "video"], "Modality must be either image or video"
        assert self.max_frames_for_video == -1 or self.frame_rate == -1, "max_frames_for_video and frame_rate cannot be provided at the same time"
        assert self.max_frames_for_video == -1 if self.frame_rate != -1 else True, "max_frames_for_video must be -1 if frame_rate > 0"
        self.image_token = "<image>"  # In case the question contains <image> token for placeholder
        self.timeout = timeout

        accelerator = Accelerator()  # This is not neccessary, it is only used to get the rank and world size
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

            self.device = self.accelerator.device

    # Function to encode the image
    def encode_image(self, image: Image):
        output_buffer = BytesIO()
        image.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()
        base64_str = base64.b64encode(byte_data).decode("utf-8")
        return base64_str

    # Function to encode the video
    def encode_video(self, video_path, for_get_frames_num=-1, frame_rate=-1):
        assert for_get_frames_num != -1 or frame_rate != -1, "Either for_get_frames_num or frame_rate must be provided"
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        if for_get_frames_num != -1:
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, for_get_frames_num, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
        else:
            frame_idx = range(0, total_frame_num, frame_rate)
        frames = vr.get_batch(frame_idx).asnumpy()

        base64_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            output_buffer = BytesIO()
            img.save(output_buffer, format="PNG")
            byte_data = output_buffer.getvalue()
            base64_str = base64.b64encode(byte_data).decode("utf-8")
            base64_frames.append(base64_str)

        # Extract audio
        try:
            ar = AudioReader(video_path, sample_rate=44100, mono=False, ctx=cpu(0))
            audio = ar.get_batch(frame_idx).asnumpy()
            audio_buffer = BytesIO()
            audio.save(audio_buffer, format="mp3")
            audio_byte_data = audio_buffer.getvalue()
            base64_audio = base64.b64encode(audio_byte_data).decode("utf-8")
        except Exception as e:
            eval_logger.error(f"Error extracting audio or no audio found for video: {video_path}")
            base64_audio = None

        return base64_frames, base64_audio

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # encode, pad, and truncate contexts for this batch
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            imgs = []  # multiple images or frames for video
            for visual in visuals:
                if self.modality == "image":
                    img = self.encode_image(visual)
                    imgs.append(img)
                elif self.modality == "video":
                    frames, audio = self.encode_video(visual, self.max_frames_for_video, self.frame_rate)  # FIXME: I am not sure how to put audio information into query, please modify this
                    imgs.extend(frames)

            payload = {"model": self.model_version, "messages": []}
            response_json = {"role": "user", "content": []}
            # When there is no image token in the context, append the image to the text
            if self.image_token not in contexts:
                payload["messages"].append(deepcopy(response_json))
                payload["messages"][0]["content"].append({"type": "text", "text": contexts})
                for img in imgs:
                    payload["messages"][0]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})
            else:
                contexts = contexts.split(self.image_token)
                for idx, img in enumerate(imgs):
                    payload["messages"].append(deepcopy(response_json))
                    payload["messages"][idx]["content"].append({"type": "text", "text": contexts[idx]})
                    payload["messages"][idx]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})

                # If n image tokens are in the contexts
                # contexts will be splitted into n+1 chunks
                # Manually add it into the payload
                payload["messages"].append(deepcopy(response_json))
                payload["messages"][-1]["content"].append({"type": "text", "text": contexts[-1]})

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0

            payload["max_tokens"] = gen_kwargs["max_new_tokens"]
            payload["temperature"] = gen_kwargs["temperature"]

            for attempt in range(5):
                try:
                    response = url_requests.post(API_URL, headers=headers, json=payload, timeout=self.timeout)
                    response_data = response.json()

                    content = response_data["choices"][0]["message"]["content"].strip()
                    break  # If successful, break out of the loop

                except Exception as e:
                    eval_logger.info(f"Attempt {attempt + 1} failed with error: {str(e)}")
                    if attempt < 5 - 1:  # If we have retries left, sleep and then continue to next attempt
                        time.sleep(NUM_SECONDS_TO_SLEEP)
                    else:  # If this was the last attempt, log and return empty
                        eval_logger.error(f"All 5 attempts failed. Last error message: {str(e)}")
                        content = ""
            res.append(content)
            pbar.update(1)
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, "Gemini API not support"
