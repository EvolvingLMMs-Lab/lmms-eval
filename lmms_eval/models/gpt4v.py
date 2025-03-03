import base64
import json
import os
import time
from copy import deepcopy
from io import BytesIO
from typing import List, Tuple, Union

import numpy as np
import requests as url_requests
from accelerate import Accelerator, DistributedType
from openai import AzureOpenAI, OpenAI
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

try:
    from decord import VideoReader, cpu
except ImportError:
    pass

from loguru import logger as eval_logger
from PIL import Image

API_TYPE = os.getenv("API_TYPE", "openai")
NUM_SECONDS_TO_SLEEP = 10
if API_TYPE == "openai":
    API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")

elif API_TYPE == "azure":
    API_URL = os.getenv("AZURE_ENDPOINT", "https://api.cognitive.microsoft.com/sts/v1.0/issueToken")
    API_KEY = os.getenv("AZURE_API_KEY", "YOUR_API_KEY")
    API_VERSION = os.getenv("AZURE_API_VERSION", "2023-07-01-preview")


@register_model("gpt4v")
class GPT4V(lmms):
    def __init__(
        self,
        model_version: str = "gpt-4-vision-preview",
        modality: str = "video",
        max_frames_num: int = 10,
        timeout: int = 120,
        continual_mode: bool = False,
        response_persistent_folder: str = None,
        max_size_in_mb: int = 20,
        **kwargs,
    ) -> None:
        super().__init__()
        # Manually set a image token for GPT4V so that we can search for it
        # and split the text and image
        # Here we just use the same token as llava for convenient
        self.model_version = model_version
        self.modality = modality
        self.max_frames_num = max_frames_num
        self.image_token = "<image>"
        self.timeout = timeout
        self.continual_mode = continual_mode
        if self.continual_mode:
            if response_persistent_folder is None:
                raise ValueError("Continual mode requires a persistent path for the response. Please provide a valid path.")

            os.makedirs(response_persistent_folder, exist_ok=True)
            self.response_persistent_folder = response_persistent_folder
            self.response_persistent_file = os.path.join(self.response_persistent_folder, f"{self.model_version}_response.json")

            if os.path.exists(self.response_persistent_file):
                with open(self.response_persistent_file, "r") as f:
                    self.response_cache = json.load(f)
                self.cache_mode = "resume"
            else:
                self.response_cache = {}
                self.cache_mode = "start"

        if API_TYPE == "openai":
            self.client = OpenAI(api_key=API_KEY)
        elif API_TYPE == "azure":
            self.client = AzureOpenAI(api_key=API_KEY, azure_endpoint=API_URL, api_version=API_VERSION)

        accelerator = Accelerator()
        # assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        self.max_size_in_mb = max_size_in_mb
        self.device = self.accelerator.device

    # Function to encode the image
    def encode_image(self, image: Union[Image.Image, str]):
        max_size = self.max_size_in_mb * 1024 * 1024  # 20MB in bytes
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.copy()

        output_buffer = BytesIO()
        img.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()

        # If image is too large, resize it while maintaining aspect ratio
        while len(byte_data) > max_size and img.size[0] > 100 and img.size[1] > 100:
            new_size = (int(img.size[0] * 0.75), int(img.size[1] * 0.75))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

            output_buffer = BytesIO()
            img.save(output_buffer, format="PNG")
            byte_data = output_buffer.getvalue()

        base64_str = base64.b64encode(byte_data).decode("utf-8")
        return base64_str

    # Function to encode the video
    def encode_video(self, video_path, for_get_frames_num):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, for_get_frames_num, dtype=int)

        # Ensure the last frame is included
        if total_frame_num - 1 not in uniform_sampled_frames:
            uniform_sampled_frames = np.append(uniform_sampled_frames, total_frame_num - 1)

        frame_idx = uniform_sampled_frames.tolist()
        frames = vr.get_batch(frame_idx).asnumpy()

        base64_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            output_buffer = BytesIO()
            img.save(output_buffer, format="PNG")
            byte_data = output_buffer.getvalue()
            base64_str = base64.b64encode(byte_data).decode("utf-8")
            base64_frames.append(base64_str)

        return base64_frames

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
            if self.continual_mode is True and self.cache_mode == "resume":
                doc_uuid = f"{task}___{split}___{doc_id}"
                if doc_uuid in self.response_cache:
                    response_text = self.response_cache[doc_uuid]
                    if response_text:
                        res.append(response_text)
                        pbar.update(1)
                        continue

            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            if None in visuals:
                visuals = []
                imgs = []
            else:
                visuals = self.flatten(visuals)
                imgs = []  # multiple images or frames for video
                for visual in visuals:
                    if isinstance(visual, str) and (".mp4" in visual or ".avi" in visual or ".mov" in visual or ".flv" in visual or ".wmv" in visual):
                        frames = self.encode_video(visual, self.max_frames_num)
                        imgs.extend(frames)
                    elif isinstance(visual, str) and (".jpg" in visual or ".jpeg" in visual or ".png" in visual or ".gif" in visual or ".bmp" in visual or ".tiff" in visual or ".webp" in visual):
                        img = self.encode_image(visual)
                        imgs.append(img)
                    elif isinstance(visual, Image.Image):
                        img = self.encode_image(visual)
                        imgs.append(img)

            payload = {"messages": []}
            payload["model"] = self.model_version

            payload["messages"].append({"role": "user", "content": []})
            payload["messages"][0]["content"].append({"type": "text", "text": contexts})
            for img in imgs:
                payload["messages"][0]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if gen_kwargs["max_new_tokens"] > 4096:
                gen_kwargs["max_new_tokens"] = 4096
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            payload["max_tokens"] = gen_kwargs["max_new_tokens"]
            payload["temperature"] = gen_kwargs["temperature"]

            MAX_RETRIES = 5
            for attempt in range(MAX_RETRIES):
                try:
                    response = self.client.chat.completions.create(**payload)
                    response_text = response.choices[0].message.content
                    break  # If successful, break out of the loop

                except Exception as e:
                    error_msg = str(e)
                    eval_logger.info(f"Attempt {attempt + 1}/{MAX_RETRIES} failed with error: {error_msg}")

                    # On last attempt, log error and set empty response
                    if attempt == MAX_RETRIES - 1:
                        eval_logger.error(f"All {MAX_RETRIES} attempts failed. Last error: {error_msg}")
                        response_text = ""
                    else:
                        time.sleep(NUM_SECONDS_TO_SLEEP)

            res.append(response_text)
            pbar.update(1)

            if self.continual_mode is True:  # Cache the response
                doc_uuid = f"{task}___{split}___{doc_id}"
                self.response_cache[doc_uuid] = response_text
                with open(self.response_persistent_file, "w") as f:
                    json.dump(self.response_cache, f)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for GPT4V")

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, "GPT4V not support"
